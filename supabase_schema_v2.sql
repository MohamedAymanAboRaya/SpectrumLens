-- ============================================================
-- SpectrumLens: Supabase Schema v2 — Dual Text Fields
-- Adds original_text + normalized_text columns.
-- FTS trigger updated to use normalized_text.
-- Backward compatible: content column still exists.
-- ============================================================

-- ─── 1. Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;


-- ─── 2. Add new columns (safe migration — idempotent) ───────────────────────────
ALTER TABLE spectrumlens_clinical_chunks
    ADD COLUMN IF NOT EXISTS original_text TEXT;

-- Backfill original_text from content for existing rows
UPDATE spectrumlens_clinical_chunks
SET original_text = content
WHERE original_text IS NULL;

-- Ensure normalized_text column exists (may already exist from v1)
ALTER TABLE spectrumlens_clinical_chunks
    ADD COLUMN IF NOT EXISTS normalized_text TEXT;

ALTER TABLE spectrumlens_clinical_chunks
    ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en';


-- ─── 3. Indexes (idempotent) ────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_spectrumlens_embedding
    ON spectrumlens_clinical_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 15);

CREATE INDEX IF NOT EXISTS idx_spectrumlens_fts
    ON spectrumlens_clinical_chunks
    USING gin(fts_vector);

CREATE INDEX IF NOT EXISTS idx_spectrumlens_doc_name
    ON spectrumlens_clinical_chunks (document_name);

CREATE INDEX IF NOT EXISTS idx_spectrumlens_language
    ON spectrumlens_clinical_chunks (language);


-- ─── 4. Updated FTS Trigger — uses normalized_text ──────────────────────────────
CREATE OR REPLACE FUNCTION update_fts_vector()
RETURNS TRIGGER AS $$
DECLARE
    text_config TEXT;
    norm_text   TEXT;
BEGIN
    -- Use normalized_text if available, fall back to original_text, then content
    norm_text := COALESCE(NEW.normalized_text, NEW.original_text, NEW.content, '');

    text_config := CASE
        WHEN NEW.language = 'ar'    THEN 'simple'
        WHEN NEW.language = 'en'    THEN 'english'
        ELSE                             'simple'
    END;

    BEGIN
        NEW.fts_vector :=
            setweight(to_tsvector(text_config, coalesce(NEW.section_title, '')), 'A') ||
            setweight(to_tsvector('simple',     coalesce(NEW.document_name, '')),  'B') ||
            setweight(to_tsvector(text_config,  coalesce(norm_text, '')),          'C');
    EXCEPTION WHEN OTHERS THEN
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
    BEFORE INSERT OR UPDATE OF content, original_text, normalized_text, section_title, document_name, language
    ON spectrumlens_clinical_chunks
    FOR EACH ROW EXECUTE FUNCTION update_fts_vector();


-- ─── 5. Semantic Search ─────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION match_clinical_chunks(
    query_embedding VECTOR(1024),
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
    original_text   TEXT,
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
        c.original_text,
        c.language,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM spectrumlens_clinical_chunks c
    WHERE
        (filter = '{}'::jsonb
            OR c.document_name = filter->>'document_name')
        AND (filter = '{}'::jsonb
            OR c.language = filter->>'language'
            OR NOT filter ? 'language')
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- ─── 6. Hybrid Search with RRF ──────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION hybrid_search_clinical_chunks(
    query_text      TEXT,
    query_embedding VECTOR(1024),
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
    original_text   TEXT,
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
    semantic AS (
        SELECT
            c.chunk_id,
            c.document_name,
            c.section_title,
            c.page_number,
            c.content,
            c.original_text,
            c.language,
            1 - (c.embedding <=> query_embedding) AS score
        FROM spectrumlens_clinical_chunks c
        WHERE
            (filter = '{}'::jsonb
                OR c.document_name = filter->>'document_name')
            AND (filter = '{}'::jsonb
                OR c.language = filter->>'language'
                OR NOT filter ? 'language')
            AND 1 - (c.embedding <=> query_embedding) > match_threshold
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    bm25 AS (
        SELECT
            c.chunk_id,
            c.document_name,
            c.section_title,
            c.page_number,
            c.content,
            c.original_text,
            c.language,
            ts_rank_cd(c.fts_vector, plainto_tsquery('simple', query_text)) AS score
        FROM spectrumlens_clinical_chunks c
        WHERE
            (filter = '{}'::jsonb
                OR c.document_name = filter->>'document_name')
            AND (filter = '{}'::jsonb
                OR c.language = filter->>'language'
                OR NOT filter ? 'language')
            AND c.fts_vector @@ plainto_tsquery('simple', query_text)
        ORDER BY score DESC
        LIMIT match_count * 3
    ),
    semantic_ranked AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM semantic
    ),
    bm25_ranked AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM bm25
    ),
    fused AS (
        SELECT
            COALESCE(s.chunk_id,      b.chunk_id)      AS chunk_id,
            COALESCE(s.document_name, b.document_name) AS document_name,
            COALESCE(s.section_title, b.section_title) AS section_title,
            COALESCE(s.page_number,   b.page_number)   AS page_number,
            COALESCE(s.content,       b.content)       AS content,
            COALESCE(s.original_text, b.original_text) AS original_text,
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
        fused.original_text,
        fused.language,
        fused.semantic_score,
        fused.bm25_rank,
        fused.rrf_score
    FROM fused
    ORDER BY fused.rrf_score DESC
    LIMIT match_count;
END;
$$;


-- ─── 7. BM25-Only Search ───────────────────────────────────────────────────────
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
    original_text   TEXT,
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
        c.original_text,
        c.language,
        ts_rank_cd(c.fts_vector, plainto_tsquery('simple', query_text)) AS bm25_score
    FROM spectrumlens_clinical_chunks c
    WHERE
        (filter = '{}'::jsonb
            OR c.document_name = filter->>'document_name')
        AND (filter = '{}'::jsonb
            OR c.language = filter->>'language'
            OR NOT filter ? 'language')
        AND c.fts_vector @@ plainto_tsquery('simple', query_text)
    ORDER BY bm25_score DESC
    LIMIT match_count;
END;
$$;


-- ─── 8. Row Level Security ──────────────────────────────────────────────────────
ALTER TABLE spectrumlens_clinical_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_key_access"
    ON spectrumlens_clinical_chunks
    FOR ALL
    USING (auth.role() = 'service_role');
