"""
SpectrumLens — Day 3: Generation & Safety Layer (CRAG) — Bilingual Edition
==========================================================================
LLM Backend: Groq API — llama-3.3-70b-versatile  (free, fast, supports Arabic)
Critic uses: llama3-8b-8192 (faster for scoring)
Generator uses: llama-3.3-70b-versatile (best reasoning + Arabic support)

Bilingual update:
  • Auto-detects query language (Arabic or English)
  • Responds in the SAME language as the query
  • Arabic queries → Arabic answers with Arabic citations
  • English queries → English answers (unchanged behaviour)
  • Critic evaluates relevance regardless of query language

Flow:
  Query
    │
    ▼
  Language Detection
    │
    ▼
  Hybrid Retriever (top-20, Semantic + BM25 + RRF)
    │
    ▼
  Cross-Encoder Reranker (top-5)
    │
    ▼
  Groq Critic (llama3-8b-8192) — evaluates each chunk
    │
    ├── SUFFICIENT  ──► Groq Generator (llama-3.3-70b-versatile)  ──► CitedResponse
    └── INSUFFICIENT ──► Safe Failure

Golden Rule: "A fluent answer does NOT mean a safe answer."
"""

import os
import json
import logging
import argparse
import textwrap
from enum import Enum
from typing import List, Dict, Any, Optional

from groq import Groq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from day2_retrieval import VectorDBManager, ClinicalRetriever, ClinicalQuery
from reranker import ClinicalReranker, RankedChunk
from arabic_preprocessor import detect_language

load_dotenv()

import re as _re

def _strip_think_and_parse(raw: str) -> str:
    """Strip <think>...</think> blocks from LLM output and return cleaned text."""
    return _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SpectrumLens-Generation")

# ─── Constants ───────────────────────────────────────────────────────────────────
SCOPE_MODEL      = "qwen/qwen3.6-27b"              # fastest Groq model
CRITIC_MODEL     = "qwen/qwen3.6-27b"              # fast for scoring
GENERATOR_MODEL  = "qwen/qwen3.6-27b"              # fastest generation
RETRIEVAL_CANDIDATE_K = 20
RERANK_TOP_N          = 5
EVAL_THRESHOLD        = 6.0    # 0-10 scale: chunk must average ≥ 6 to pass
MIN_RELEVANT_CHUNKS   = 2      # at least 2 chunks with score ≥ 6 required
MAX_OUTPUT_TOKENS     = 1500


# ─── Data Models ─────────────────────────────────────────────────────────────────
class EvalVerdict(str, Enum):
    SUFFICIENT   = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class ChunkEvaluation(BaseModel):
    chunk_idx:  int    = Field(ge=0, description="Zero-based index of the chunk")
    score:      int    = Field(ge=0, le=10, description="Relevance score 0-10")
    reasoning:  str    = Field(description="One concise sentence explaining the score")


class ContextEvalReport(BaseModel):
    verdict:           EvalVerdict
    chunk_evaluations: List[ChunkEvaluation]
    mean_relevance:    float
    relevant_count:    int
    evaluator_notes:   str


class ScopeLevel(str, Enum):
    ALLOWED      = "ALLOWED"
    NEEDS_CAUTION = "NEEDS_CAUTION"
    REFUSE       = "REFUSE"


class ScopeCheckResult(BaseModel):
    scope_level: ScopeLevel
    query:       str


class Citation(BaseModel):
    document_name: str
    section_title: str
    page_number:   str


class ClinicalResponse(BaseModel):
    query:               str
    verdict:             EvalVerdict
    answer:              Optional[str] = None
    citations:           List[Citation] = Field(default_factory=list)
    safe_failure_reason: Optional[str] = None
    context_report:      ContextEvalReport
    rerank_scores:       List[Dict[str, Any]] = Field(default_factory=list)
    query_language:      str = "en"    # "ar" | "en" | "mixed"
    scope_level:         ScopeLevel = ScopeLevel.ALLOWED


# ─── Scope Checker (Pre-Retrieval OOS Detection) ─────────────────────────────────
class ScopeChecker:
    """
    Pre-retrieval ternary scope check: classifies the query into one of three levels.
    Runs BEFORE retrieval to save resources and prevent irrelevant or dangerous answers.
    Uses the fast critic model for low latency (~200ms).
    """

    SYSTEM_PROMPT = """You are a ternary scope classifier for an ASD clinical decision support system.

Classify the user query into exactly one of three levels:

ALLOWED — Clinical guideline questions. Safe to answer from guidelines.
Examples: "What are the DSM-5 criteria for ASD?", "ASD screening age recommendations",
"What is the AAP guideline for autism screening?", "risperidone dosage for ASD irritability in guidelines",
"What interventions does NICE recommend for ASD?"

NEEDS_CAUTION — Patient-specific or personal scenario questions. Can be answered with general
guideline information but requires a clear disclaimer that this is NOT personalized medical advice.
Examples: "My child shows signs of autism, what should I do?", "Should I get my son evaluated for ASD?",
"My 3-year-old isn't speaking — could it be autism?", "I think I might have autism, what tests are there?",
"Should I start ABA therapy for my child?"

REFUSE — Dangerous, inappropriate, or out-of-scope requests. Must NOT be answered.
Examples: "What dose of risperidone should I give my child?", "Can you diagnose my child with autism?",
"What is the cure for autism?", "I have autism, what medication should I take without seeing a doctor?",
"Give me a treatment plan for my patient", completely non-ASD medical questions, non-medical topics.

Rules:
- REFUSE takes priority: if the query asks for specific patient dosing, diagnosis, or treatment plans, always REFUSE
- NEEDS_CAUTION applies when the query is personal/patient-specific but asks for general guideline info
- ALLOWED applies to general clinical knowledge questions that don't target a specific individual
- Out-of-scope queries (non-ASD, non-medical) should be REFUSEd

Respond with ONLY valid JSON: {"scope_level": "ALLOWED"} or {"scope_level": "NEEDS_CAUTION"} or {"scope_level": "REFUSE"}\n\n/no_think"""

    def __init__(self, client: Groq):
        self.client = client

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def check(self, query: str) -> ScopeCheckResult:
        logger.info(f"Scope check: '{query[:80]}…'")
        resp = self.client.chat.completions.create(
            model=SCOPE_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = resp.choices[0].message.content.strip()
        raw = _strip_think_and_parse(raw)
        try:
            parsed = json.loads(raw)
            level_str = parsed.get("scope_level", "REFUSE").upper()
            scope_level = ScopeLevel(level_str) if level_str in ScopeLevel.__members__.values() else ScopeLevel.REFUSE
        except (json.JSONDecodeError, KeyError, ValueError):
            scope_level = ScopeLevel.REFUSE  # Fail closed for safety
        logger.info(f"Scope check result: {scope_level.value}")
        return ScopeCheckResult(scope_level=scope_level, query=query)


# ─── Critic Agent (Groq) ─────────────────────────────────────────────────────────
class ContextEvaluator:
    """
    Uses llama3-8b-8192 on Groq as the Critic.
    Fast, deterministic (temperature=0), strict on clinical relevance.
    Language-agnostic: evaluates chunk relevance regardless of query language.
    """

    SYSTEM_PROMPT = """You are a clinical evidence evaluator for an ASD (Autism Spectrum Disorder) decision support system.

TASK: Score each retrieved chunk for relevance to the clinical query.

SCORING RUBRIC (0-10):
- 9-10: Directly answers the query with specific clinical data (thresholds, criteria, drug names, ages, screening tools)
- 7-8: Strongly relevant — provides supporting evidence or partial answer with clinical specifics
- 5-6: Topically related but lacks direct answer to this specific query
- 3-4: Tangentially related (same disease area, different question)
- 1-2: Weakly related (mentions ASD but wrong context)
- 0: Not relevant

AUTHORITY TIERS (higher = better):
1. Official guidelines (AAP, NICE CG128, DSM-5, ICD-11, WHO)
2. Peer-reviewed research and consensus statements
3. Clinical toolkits and educational materials

CONSIDER for each chunk:
- Clinical relevance: Does it directly address the specific question?
- Evidence specificity: Contains exact data (ages, scores, drug names) vs. vague generalities?
- Guideline authority: Official guidelines > reviews > toolkits

ASD CLINICAL CONTEXT: This system covers ASD screening (M-CHAT-R/F, AAP), diagnosis (DSM-5, ICD-11), interventions (ABA, EIBI), medications (risperidone, aripiprazole), guidelines (NICE CG128, AAP 2020), and comorbidities.

OUTPUT FORMAT (JSON only):
{"scores": [{"chunk_idx": 0, "score": 8, "reasoning": "Contains AAP screening age recommendation"}]}

RULES:
- Score based on clinical relevance, not writing quality
- Be strict: a chunk about "ASD prevalence" is NOT relevant to "ASD screening age"
- Every chunk gets exactly one score (0-10) and one reasoning sentence

/no_think"""

    def __init__(self, client: Groq):
        self.client = client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def evaluate(self, query: str, chunks: List[RankedChunk]) -> ContextEvalReport:
        if not chunks:
            return ContextEvalReport(
                verdict=EvalVerdict.INSUFFICIENT,
                chunk_evaluations=[], mean_relevance=0.0, relevant_count=0,
                evaluator_notes="No chunks retrieved — Safe Failure.",
            )

        chunks_text = "\n\n".join(
            f"[CHUNK {i+1}]\nchunk_id: {c.chunk_id}\ndocument: {c.document_name}\n"
            f"section: {c.section_title}\nrerank_score: {c.rerank_score:.4f}\n"
            f"content: {c.content[:500]}"
            for i, c in enumerate(chunks)
        )

        logger.info(f"Critic evaluating {len(chunks)} chunk(s) via Groq …")

        resp = self.client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"CLINICAL QUERY:\n{query}\n\nCHUNKS TO EVALUATE:\n{chunks_text}"},
            ],
            temperature=0.0,
            max_tokens=1200,
        )

        raw = resp.choices[0].message.content.strip()
        raw = _strip_think_and_parse(raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Critic JSON parse error: {e}")
            return ContextEvalReport(
                verdict=EvalVerdict.INSUFFICIENT,
                chunk_evaluations=[], mean_relevance=0.0, relevant_count=0,
                evaluator_notes="Evaluator parse failure — Safe Failure.",
            )

        evals    = [ChunkEvaluation(**e) for e in parsed.get("scores", [])]
        relevant = [e for e in evals if e.score >= 6]
        mean_rel = sum(e.score for e in evals) / len(evals) if evals else 0.0
        verdict  = (
            EvalVerdict.SUFFICIENT
            if mean_rel >= EVAL_THRESHOLD and len(relevant) >= MIN_RELEVANT_CHUNKS
            else EvalVerdict.INSUFFICIENT
        )

        logger.info(f"Critic: {verdict.value} | mean={mean_rel:.3f} | relevant={len(relevant)}/{len(evals)}")
        return ContextEvalReport(
            verdict=verdict, chunk_evaluations=evals,
            mean_relevance=mean_rel, relevant_count=len(relevant),
            evaluator_notes=parsed.get("overall_assessment", ""),
        )


# ─── Generator (Groq) — Bilingual ────────────────────────────────────────────────
class GeminiClinicalGenerator:
    """
    Uses llama-3.3-70b-versatile on Groq for bilingual generation.
    Responds in the same language as the query (Arabic or English).
    (Class name kept for import compatibility.)
    """

    # ── English system prompt ─────────────────────────────────────────────────────
    SYSTEM_PROMPT_EN = """You are SpectrumLens, an ASD Clinical Decision Support System.

CONSTITUTION (NON-NEGOTIABLE RULES):
1. USE ONLY information from the PROVIDED CONTEXT below. Never use prior knowledge.
2. Every factual claim MUST include an inline citation using this EXACT format: 【Source N】where N matches the evidence number.
3. If the context is insufficient, respond: "⚠️ INSUFFICIENT EVIDENCE IN GUIDELINES. The retrieved guidelines do not contain sufficient evidence to answer this question reliably. Please consult the relevant clinical guideline or a qualified clinician."
4. NEVER provide patient-specific diagnosis, treatment, or dosage recommendations.
5. ALWAYS end every response with: "⚕️ This output is generated from clinical guidelines for decision support only. It does not replace professional medical judgment. Consult a qualified healthcare provider for clinical decisions."

OUTPUT STRUCTURE (use this EXACT format):

📋 **Answer**
[Direct, evidence-based answer. Every sentence MUST cite a source using 【Source N】. Example: The AAP recommends universal ASD screening at 18 months 【Source 1】. Write 3-7 sentences with inline citations throughout.]

📚 **Supporting Evidence**
• 【Source 1】 Document Name — Section Title (Page X)
• 【Source 2】 Document Name — Section Title (Page X)

🎯 **Confidence Level**: [HIGH / MEDIUM / LOW / INSUFFICIENT]
- HIGH: Multiple authoritative sources agree
- MEDIUM: Single authoritative source or moderate agreement
- LOW: Limited or conflicting evidence
- INSUFFICIENT: No relevant evidence found

⚕️ **Clinical Disclaimer**
"This output is generated from clinical guidelines for decision support only. It does not replace professional medical judgment. Consult a qualified healthcare provider for clinical decisions."

RULES:
- Every sentence in the Answer section MUST be grounded in the PROVIDED CONTEXT
- Never invent, assume, or hallucinate any clinical fact
- If you cannot fully answer from context, state what IS supported and what is NOT
- Use 【Source N】 format for inline citations — N corresponds to the evidence list above

/no_think"""

    # ── Arabic system prompt ──────────────────────────────────────────────────────
    SYSTEM_PROMPT_AR = """أنت SpectrumLens، نظام دعم القرار السريري المتخصص في اضطراب طيف التوحد (ASD).

قواعد دستورية (إلزامية):
1. استخدم فقط المعلومات من السياق المقدم أدناه. لا تستخدم معرفاً مسبقاً.
2. كل ادعاء وقائعي يجب أن يتضمن اقتباساً بالتنسيق: 【Source N】 حيث N يتطابق مع رقم الدليل.
3. إذا كان السياق غير كافٍ، استجب: "⚠️ INSUFFICIENT EVIDENCE IN GUIDELINES. الإرشادات المسترجعة لا تحتوي على أدلة كافية للإجابة على هذا السؤال بشكل موثوق. يرجى استشارة الإرشاد السريري ذي الصلة أو طبيب مؤهل."
4. لا تقدم أبداً تشخيصاً أو علاجاً أو جرعات محددة للمريض.
5. أنهِ كل استجابة بـ: "⚕️ هذا الإخراج تم إنشاؤه من إرشادات سريرية لدعم القرار فقط. لا يحل محل الحكم الطبي المهني. استشر مقدم رعاية صحة مؤهل للقرارات السريرية."

تنسيق الإجابة (استخدم هذا التنسيق بالضبط):

📋 **الإجابة**
[إجابة مباشرة مبنية على الأدلة. كل جملة يجب أن تستشهد بمصدر باستخدام 【Source N】. مثال: توصي AAP بفحص التوحد الشامل عند عمر 18 شهراً 【Source 1】. اكتب 3-7 جمل مع اقتباسات مضمنة.]

📚 **الأدلة الداعمة**
• 【Source 1】 اسم المستند — عنوان القسم (صفحة X)
• 【Source 2】 اسم المستند — عنوان القسم (صفحة X)

🎯 **مستوى الثقة**: [HIGH / MEDIUM / LOW / INSUFFICIENT]
- HIGH: مصادر رسمية متعددة تتفق
- MEDIUM: مصدر رسمي واحد أو اتفاق معقول
- LOW: أدلة محدودة أو متعارضة
- INSUFFICIENT: لم يتم العثور على أدلة ذات صلة

⚕️ **إخلاء المسؤولية السريري**
"هذا الإخراج تم إنشاؤه من إرشادات سريرية لدعم القرار فقط. لا يحل محل الحكم الطبي المهني. استشر مقدم رعاية صحة مؤهل للقرارات السريرية."

القواعد:
- كل جملة في قسم الإجابة يجب أن مستندة إلى السياق المقدم
- لا تختلق أو تفترض أو تتخيّل أي حقيقة سريرية
- إذا لم تتمكن من الإجابة الكاملة من السياق، اذكر ما هو مدعوم وما هو غير مدعوم
- الاقتباسات بالتنسيق: 【Source N】 — N يتطابق مع قائمة الأدلة أعلاه

/no_think"""

    def __init__(self, client: Groq):
        self.client = client

    def _get_system_prompt(self, lang: str) -> str:
        """Select system prompt based on detected query language."""
        if lang == "ar":
            return self.SYSTEM_PROMPT_AR
        return self.SYSTEM_PROMPT_EN

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, query: str, chunks: List[RankedChunk], lang: str = "en") -> tuple[str, List[Citation]]:
        context = "\n\n".join(
            f"[EVIDENCE {i+1}] document_name: {c.document_name}\n"
            f"section_title: {c.section_title}\npage_number: {c.page_number}\n"
            f"content: {c.content}"
            for i, c in enumerate(chunks)
        )

        system_prompt = self._get_system_prompt(lang)
        lang_instruction = (
            "\n\nIMPORTANT: The user's query is in Arabic. Respond entirely in Arabic."
            if lang == "ar" else ""
        )

        logger.info(f"Generator synthesising answer via Groq (lang={lang}) …")
        resp = self.client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt + lang_instruction},
                {"role": "user",   "content":
                    f"CLINICAL QUERY:\n{query}\n\nAPPROVED EVIDENCE:\n{context}\n\n"
                    "Generate a comprehensive cited answer following your system instructions."},
            ],
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        answer = _strip_think_and_parse(resp.choices[0].message.content.strip())

        seen: set = set()
        citations = []
        for c in chunks:
            key = (c.document_name, c.section_title, c.page_number)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    document_name=c.document_name,
                    section_title=c.section_title,
                    page_number=c.page_number,
                ))
        return answer, citations


def verify_citations(answer: str, retrieved_chunks: list) -> tuple:
    """Verify that all [SOURCE: ...] citations exist in retrieved chunks. Returns (verified_answer, missing_citations)."""
    try:
        from guardrails.citation_verifier import verify_answer
        cv = verify_answer(answer, retrieved_chunks)
        return answer, cv.missing_citations
    except Exception:
        import re
        citations = re.findall(r'\[SOURCE:\s*([^\]]+)\]', answer)
        chunk_docs = set()
        for c in retrieved_chunks:
            doc = c.get("document_name", "")
            chunk_docs.add(doc.lower())
            chunk_docs.add(doc.replace("_", " ").lower())
        missing = []
        for cite in citations:
            cite_lower = cite.strip().lower()
            if not any(cite_lower in d or d in cite_lower for d in chunk_docs):
                missing.append(cite)
        return answer, missing


# ─── CRAG Orchestrator ────────────────────────────────────────────────────────────
class CRAGOrchestrator:
    """
    Full pipeline: Retrieve (top-20) → Rerank (top-5) → Critic → Generate | SafeFail
    Bilingual: auto-detects query language and passes it to the generator.
    """

    def __init__(
        self,
        retrieval_candidate_k: int = RETRIEVAL_CANDIDATE_K,
        rerank_top_n: int = RERANK_TOP_N,
    ):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        client = Groq(api_key=api_key)

        db_manager        = VectorDBManager()
        self.retriever    = ClinicalRetriever(db_manager)
        self.reranker     = ClinicalReranker(top_n=rerank_top_n)
        self.scope_checker = ScopeChecker(client)
        self.evaluator    = ContextEvaluator(client)
        self.generator    = GeminiClinicalGenerator(client)
        self.retrieval_k  = retrieval_candidate_k

        logger.info(
            f"CRAGOrchestrator ready — critic={CRITIC_MODEL} | "
            f"generator={GENERATOR_MODEL} | embed=BGE-M3 (1024-dim) | "
            f"retrieval_k={retrieval_candidate_k} | rerank_top_n={rerank_top_n}"
        )

    def answer(
        self,
        query: str,
        target_document: Optional[str] = None,
        pre_reranked: Optional[List[RankedChunk]] = None,
    ) -> ClinicalResponse:
        logger.info(f"═══ Query ═══ '{query}'")

        # Detect query language for bilingual response
        query_lang = detect_language(query)
        logger.info(f"Query language detected: {query_lang}")

        # Step 0: Scope check — reject or flag queries BEFORE retrieval
        scope_result = self.scope_checker.check(query)
        if scope_result.scope_level == ScopeLevel.REFUSE:
            reason = (
                "This query was refused by the scope checker. It may request specific patient "
                "dosing, a personal diagnosis, a dangerous treatment plan, or fall outside the "
                "ASD clinical scope of SpectrumLens."
            )
            logger.warning(f"⛔ REFUSED query: '{query[:60]}…'")
            return ClinicalResponse(
                query=query, verdict=EvalVerdict.INSUFFICIENT,
                safe_failure_reason=reason, context_report=ContextEvalReport(
                    verdict=EvalVerdict.INSUFFICIENT, chunk_evaluations=[],
                    mean_relevance=0.0, relevant_count=0,
                    evaluator_notes="Query refused by scope check — REFUSE.",
                ),
                rerank_scores=[], query_language=query_lang,
                scope_level=ScopeLevel.REFUSE,
            )

        if pre_reranked is not None:
            # Use pre-reranked chunks from the caller (avoids double reranking)
            reranked = pre_reranked
            logger.info(f"Using {len(reranked)} pre-reranked chunks from caller.")
        else:
            # Step 1: Wide retrieval (BGE-M3 cross-lingual semantic + BM25)
            cq         = ClinicalQuery(text_query=query, target_document=target_document)
            candidates = self.retriever.retrieve_safe_context(cq, top_k=self.retrieval_k)

            # Step 2: Rerank
            reranked: List[RankedChunk] = self.reranker.rerank(query, candidates)

        rerank_scores = [
            {"chunk_id": r.chunk_id, "document_name": r.document_name,
             "page_number": r.page_number, "vector_score": r.vector_score,
             "rerank_score": r.rerank_score}
            for r in reranked
        ]

        # Step 3: Critic (language-agnostic)
        eval_report = self.evaluator.evaluate(query, reranked)

        # Step 4a: Safe Failure
        if eval_report.verdict == EvalVerdict.INSUFFICIENT:
            reason = (
                f"Evidence below clinical confidence threshold "
                f"(mean_relevance={eval_report.mean_relevance:.2f} < {EVAL_THRESHOLD}, "
                f"relevant={eval_report.relevant_count} < {MIN_RELEVANT_CHUNKS}). "
                f"{eval_report.evaluator_notes}"
            )
            logger.warning(f"⛔ Safe Failure — {reason}")
            return ClinicalResponse(
                query=query, verdict=EvalVerdict.INSUFFICIENT,
                safe_failure_reason=reason, context_report=eval_report,
                rerank_scores=rerank_scores, query_language=query_lang,
                scope_level=scope_result.scope_level,
            )

        # Step 4b: Generate in query language
        relevant = [
            c for c, ev in zip(reranked, eval_report.chunk_evaluations)
            if ev.score >= 6
        ]
        answer, citations = self.generator.generate(query, relevant, lang=query_lang)

        # Post-generation: verify citations against retrieved chunks
        chunk_dicts = [{"document_name": c.document_name} for c in relevant]
        _, missing_citations = verify_citations(answer, chunk_dicts)
        if missing_citations:
            logger.warning(f"⚠️ Missing citations not found in retrieved chunks: {missing_citations}")
        else:
            logger.info("✅ All citations verified against retrieved chunks.")

        # Handle NEEDS_CAUTION: add disclaimer and cap confidence
        if scope_result.scope_level == ScopeLevel.NEEDS_CAUTION:
            disclaimer = (
                "\n\n⚠️ This is general guideline information, not personalized medical advice. "
                "Consult a healthcare provider."
            )
            answer = answer + disclaimer
            # Cap confidence at MEDIUM by replacing HIGH with MEDIUM
            answer = answer.replace("Confidence Level**: HIGH", "Confidence Level**: MEDIUM")
            answer = answer.replace("مستوى الثقة**: HIGH", "مستوى الثقة**: MEDIUM")
            logger.info("⚠️ NEEDS_CAUTION — disclaimer added, confidence capped at MEDIUM")

        logger.info("✅ Clinical answer generated.")
        return ClinicalResponse(
            query=query, verdict=EvalVerdict.SUFFICIENT,
            answer=answer, citations=citations,
            context_report=eval_report, rerank_scores=rerank_scores,
            query_language=query_lang, scope_level=scope_result.scope_level,
        )


# ─── Pretty print ─────────────────────────────────────────────────────────────────
def print_response(r: ClinicalResponse) -> None:
    w = 76
    print(f"\n{'═'*w}")
    print(f"  QUERY  : {r.query}")
    print(f"  LANG   : {r.query_language}")
    print(f"  SCOPE  : {r.scope_level.value}")
    print(f"  VERDICT: {r.verdict.value}")
    print(f"{'═'*w}")

    if r.verdict == EvalVerdict.INSUFFICIENT:
        print("\n⛔  SAFE FAILURE — answer withheld for clinical safety.")
        print(f"\n   {r.safe_failure_reason}")
    else:
        print("\n📋  CLINICAL ANSWER:\n")
        for line in r.answer.split("\n"):
            print(f"  {line}")
        print("\n📚  CITATIONS:")
        for i, c in enumerate(r.citations, 1):
            print(f"  [{i}] {c.document_name} | {c.section_title} | p.{c.page_number}")

    if r.rerank_scores:
        print("\n🏆  RERANK SCORES:")
        for rs in r.rerank_scores:
            print(f"   [{rs['rerank_score']:+.4f}] vec={rs['vector_score']:.3f} "
                  f"| {rs['document_name']} p.{rs['page_number']}")

    rep = r.context_report
    print(f"\n🔍  CRITIC: mean_score={rep.mean_relevance:.1f}/10 | "
          f"relevant={rep.relevant_count} | {rep.evaluator_notes}")
    print(f"{'─'*w}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SpectrumLens CRAG — Bilingual Edition")
    p.add_argument("--query",         type=str)
    p.add_argument("--doc",           type=str, default=None)
    p.add_argument("--retrieval-k",   type=int, default=RETRIEVAL_CANDIDATE_K)
    p.add_argument("--rerank-top-n",  type=int, default=RERANK_TOP_N)
    p.add_argument("--interactive",   action="store_true")
    p.add_argument("--output-json",   type=str, default=None)
    args = p.parse_args()

    orch = CRAGOrchestrator(
        retrieval_candidate_k=args.retrieval_k,
        rerank_top_n=args.rerank_top_n,
    )

    if args.query:
        resp = orch.answer(query=args.query, target_document=args.doc)
        print_response(resp)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(resp.model_dump(), f, indent=2, ensure_ascii=False)

    elif args.interactive:
        print("\n🔬  SpectrumLens — Interactive Bilingual Mode (type 'quit' to exit)\n")
        print("    Supports Arabic and English queries.\n")
        while True:
            try:
                q = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"quit", "exit", "q"}:
                break
            print_response(orch.answer(query=q))
    else:
        p.print_help()
