-- ============================================================
-- SpectrumLens: Supabase Schema — Multilingual Hybrid Search Edition
-- Embedding: BAAI/bge-m3 (1024-dim, 100+ languages including Arabic)
-- Run this in the Supabase SQL Editor ONCE before Day 2 upload
-- ============================================================
--
-- BREAKING CHANGE from previous version:
--   • VECTOR(384) → VECTOR(1024)  (BGE-M3 dimension)
--   • Added normalized_text column (for bilingual BM25/FTS retrieval)
--   • Added language column ("ar" | "en" | "mixed")
--   • FTS trigger now handles both Arabic and English using 'simple' dict
--
-- Migration from old schema:
--   1. DROP TABLE spectrumlens_clinical_chunks CASCADE;
--   2. Run this file
--   3. Re-run: python day2_retrieval.py --upload
-- ============================================================

-- ─── 1. Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS unaccent;    -- for robust FTS on medical abbreviations


-- ─── 2. Clinical Chunks Table ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS spectrumlens_clinical_chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        TEXT UNIQUE NOT NULL,
    document_name   TEXT NOT NULL,
    section_title   TEXT,
    page_number     TEXT,

    -- Content fields (dual representation per guidelines)
    content         TEXT NOT NULL,         -- original text — for citations/display
    normalized_text TEXT,                  -- Arabic/English normalized — for retrieval
    language        TEXT DEFAULT 'en',     -- "ar" | "en" | "mixed" | "unknown"

    -- BGE-M3: 1024 dimensions (was 384 for bge-small-en-v1.5)
    embedding       VECTOR(1024),

    -- BM25/Full-Text Search column (auto-maintained by trigger below)
    -- Uses 'simple' dictionary for language-agnostic matching (Arabic + English)
    fts_vector      TSVECTOR,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ─── 3. Indexes ──────────────────────────────────────────────────────────────────

-- Semantic: IVFFlat ANN index on 1024-dim vectors
-- lists = sqrt(n_rows) is a good starting point; adjust after >1000 rows.
-- For ~500 rows, lists=15 gives reasonable recall. Increase to 100+ for >10k rows.
CREATE INDEX IF NOT EXISTS idx_spectrumlens_embedding
    ON spectrumlens_clinical_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 15);

-- BM25/Full-Text: GIN index on the tsvector column (very fast keyword lookup)
CREATE INDEX IF NOT EXISTS idx_spectrumlens_fts
    ON spectrumlens_clinical_chunks
    USING gin(fts_vector);

-- Metadata filter indexes
CREATE INDEX IF NOT EXISTS idx_spectrumlens_doc_name
    ON spectrumlens_clinical_chunks (document_name);

CREATE INDEX IF NOT EXISTS idx_spectrumlens_language
    ON spectrumlens_clinical_chunks (language);


-- ─── 4. Bilingual FTS Auto-Update Trigger ────────────────────────────────────────
-- Keeps fts_vector in sync whenever content is inserted or updated.
--
-- Bilingual strategy:
--   Arabic text  → 'arabic'  dictionary (where available in PostgreSQL)
--   English text → 'english' dictionary
--   Mixed/other  → 'simple'  dictionary (language-agnostic, safe fallback)
--
-- Note: PostgreSQL's 'arabic' text search dictionary may need the
-- postgresql-contrib package. If unavailable, 'simple' is used as a
-- safe universal fallback for all languages.
--
-- We build FTS from normalized_text (for BM25 matching) and add metadata
-- weight from section_title and document_name.

CREATE OR REPLACE FUNCTION update_fts_vector()
RETURNS TRIGGER AS $$
DECLARE
    text_config TEXT;
    norm_text   TEXT;
BEGIN
    -- Use normalized_text if available, fall back to content
    norm_text := COALESCE(NEW.normalized_text, NEW.content, '');

    -- Select dictionary based on language field
    text_config := CASE
        WHEN NEW.language = 'ar'    THEN 'simple'    -- 'arabic' if available
        WHEN NEW.language = 'en'    THEN 'english'
        ELSE                             'simple'    -- mixed / unknown
    END;

    -- Build weighted tsvector:
    --   A = section_title  (highest weight — medical sections are discriminative)
    --   B = document_name
    --   C = normalized content (main body for keyword matching)
    BEGIN
        NEW.fts_vector :=
            setweight(to_tsvector(text_config, coalesce(NEW.section_title, '')), 'A') ||
            setweight(to_tsvector('simple',     coalesce(NEW.document_name, '')),  'B') ||
            setweight(to_tsvector(text_config,  coalesce(norm_text, '')),          'C');
    EXCEPTION WHEN OTHERS THEN
        -- Final fallback: use 'simple' for any unsupported language config
        NEW.fts_vector :=
            setweight(to_tsvector('simple', coalesce(NEW.section_title, '')), 'A') ||
            setweight(to_tsvector('simple', coalesce(NEW.document_name, '')), 'B') ||
            setweight(to_tsvector('simple', coalesce(norm_text, '')),         'C');
    END;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_update_fts ON spectrumlens_clinical_chunks;
CREATE TRIGGER trig_update_fts
    BEFORE INSERT OR UPDATE OF content, normalized_text, section_title, document_name, language
    ON spectrumlens_clinical_chunks
    FOR EACH ROW EXECUTE FUNCTION update_fts_vector();


-- ─── 5. Semantic-Only Search (1024-dim BGE-M3) ───────────────────────────────────
CREATE OR REPLACE FUNCTION match_clinical_chunks(
    query_embedding VECTOR(1024),    -- 1024-dim for BGE-M3
    match_threshold FLOAT,
    match_count     INT,
    filter          JSONB DEFAULT '{}'
)
RETURNS TABLE (
    chunk_id        TEXT,
    document_name   TEXT,
    section_title   TEXT,
    page_number     TEXT,
    content         TEXT,
    language        TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        c.document_name,
        c.section_title,
        c.page_number,
        c.content,
        c.language,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM spectrumlens_clinical_chunks c
    WHERE
        (filter = '{}'::jsonb OR c.document_name = filter->>'document_name')
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- ─── 6. Hybrid Search with Reciprocal Rank Fusion (RRF) ──────────────────────────
--
-- BM25 leg uses 'simple' dictionary — language-agnostic, works for
-- both Arabic and English queries without configuration changes.
-- The 'simple' dictionary does not stem or stop-word filter, so Arabic
-- normalized forms from normalized_text are matched exactly.
--
-- RRF score = Σ  1 / (k + rank_i)   where k=60 is the standard smoothing constant.

CREATE OR REPLACE FUNCTION hybrid_search_clinical_chunks(
    query_text      TEXT,               -- raw user query (for BM25)
    query_embedding VECTOR(1024),       -- BGE-M3 embedded query (for semantic)
    match_threshold FLOAT  DEFAULT 0.35,
    match_count     INT    DEFAULT 20,
    rrf_k           INT    DEFAULT 60,
    filter          JSONB  DEFAULT '{}'
)
RETURNS TABLE (
    chunk_id        TEXT,
    document_name   TEXT,
    section_title   TEXT,
    page_number     TEXT,
    content         TEXT,
    language        TEXT,
    semantic_score  FLOAT,
    bm25_rank       BIGINT,
    rrf_score       FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH
    -- ── Leg 1: Semantic retrieval (BGE-M3, cross-lingual) ─────────────────────
    semantic AS (
        SELECT
            c.chunk_id,
            c.document_name,
            c.section_title,
            c.page_number,
            c.content,
            c.language,
            1 - (c.embedding <=> query_embedding) AS score
        FROM spectrumlens_clinical_chunks c
        WHERE
            (filter = '{}'::jsonb OR c.document_name = filter->>'document_name')
            AND 1 - (c.embedding <=> query_embedding) > match_threshold
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    -- ── Leg 2: BM25 / Full-Text retrieval (language-agnostic 'simple' dict) ──
    -- plainto_tsquery('simple', ...) handles Arabic + English queries safely
    bm25 AS (
        SELECT
            c.chunk_id,
            c.document_name,
            c.section_title,
            c.page_number,
            c.content,
            c.language,
            ts_rank_cd(c.fts_vector, plainto_tsquery('simple', query_text)) AS score
        FROM spectrumlens_clinical_chunks c
        WHERE
            (filter = '{}'::jsonb OR c.document_name = filter->>'document_name')
            AND c.fts_vector @@ plainto_tsquery('simple', query_text)
        ORDER BY score DESC
        LIMIT match_count * 3
    ),
    -- ── Assign per-leg ranks ───────────────────────────────────────────────────
    semantic_ranked AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM semantic
    ),
    bm25_ranked AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM bm25
    ),
    -- ── RRF Fusion ────────────────────────────────────────────────────────────
    fused AS (
        SELECT
            COALESCE(s.chunk_id,      b.chunk_id)      AS chunk_id,
            COALESCE(s.document_name, b.document_name) AS document_name,
            COALESCE(s.section_title, b.section_title) AS section_title,
            COALESCE(s.page_number,   b.page_number)   AS page_number,
            COALESCE(s.content,       b.content)       AS content,
            COALESCE(s.language,      b.language)      AS language,
            COALESCE(s.score, 0.0)                     AS semantic_score,
            COALESCE(b.rank, match_count * 3 + 1)      AS bm25_rank,
            (
                COALESCE(1.0 / (rrf_k + s.rank), 0.0) +
                COALESCE(1.0 / (rrf_k + b.rank), 0.0)
            ) AS rrf_score
        FROM semantic_ranked s
        FULL OUTER JOIN bm25_ranked b ON s.chunk_id = b.chunk_id
    )
    SELECT
        fused.chunk_id,
        fused.document_name,
        fused.section_title,
        fused.page_number,
        fused.content,
        fused.language,
        fused.semantic_score,
        fused.bm25_rank,
        fused.rrf_score
    FROM fused
    ORDER BY fused.rrf_score DESC
    LIMIT match_count;
END;
$$;


-- ─── 7. Row Level Security ────────────────────────────────────────────────────────
ALTER TABLE spectrumlens_clinical_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_key_access"
    ON spectrumlens_clinical_chunks
    FOR ALL
    USING (auth.role() = 'service_role');


-- ─── 8. Backfill FTS for existing rows ───────────────────────────────────────────
-- If you already have data and need to rebuild fts_vector, run:
-- UPDATE spectrumlens_clinical_chunks SET content = content;


-- ─── 9. BM25-Only Search with Rank ──────────────────────────────────────────────
-- Returns ts_rank_cd for display in the Evidence Panel.
CREATE OR REPLACE FUNCTION bm25_search_clinical_chunks(
    query_text  TEXT,
    match_count INT    DEFAULT 20,
    filter      JSONB  DEFAULT '{}'
)
RETURNS TABLE (
    chunk_id        TEXT,
    document_name   TEXT,
    section_title   TEXT,
    page_number     TEXT,
    content         TEXT,
    language        TEXT,
    bm25_score      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        c.document_name,
        c.section_title,
        c.page_number,
        c.content,
        c.language,
        ts_rank_cd(c.fts_vector, plainto_tsquery('simple', query_text)) AS bm25_score
    FROM spectrumlens_clinical_chunks c
    WHERE
        (filter = '{}'::jsonb OR c.document_name = filter->>'document_name')
        AND c.fts_vector @@ plainto_tsquery('simple', query_text)
    ORDER BY bm25_score DESC
    LIMIT match_count;
END;
$$;
