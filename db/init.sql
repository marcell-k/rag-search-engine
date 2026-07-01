CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy matching later

CREATE TABLE chunks (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        VARCHAR(64)  UNIQUE NOT NULL,   -- sha256(cik + chunk_index)
    chunk_index     INTEGER      NOT NULL,

    -- Entity
    cik             VARCHAR(20)  NOT NULL,
    company_name    TEXT         NOT NULL,
    company_ticker  TEXT[],

    -- Filing
    filing_type     VARCHAR(20)  NOT NULL,
    period_of_report DATE,
    period_type     VARCHAR(10),
    accession_number VARCHAR(50) NOT NULL,
    is_amendment    BOOLEAN      NOT NULL DEFAULT FALSE,

    -- Hierarchy
    sec_item        VARCHAR(50),
    sec_title       TEXT,
    subsection_path TEXT[],
    note_number     VARCHAR(20),

    -- Content
    content         TEXT         NOT NULL,
    token_count     INTEGER      NOT NULL,

    -- Flags
    is_table        BOOLEAN      NOT NULL DEFAULT FALSE,
    is_footnote     BOOLEAN      NOT NULL DEFAULT FALSE,
    is_boilerplate  BOOLEAN      NOT NULL DEFAULT FALSE,
    contains_numbers BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Topics
    topics          TEXT[],

    -- Vectors + FTS
    embedding       VECTOR(384),            -- all-MiniLM-L6-v2; change dim if swapping model
    fts_vector      TSVECTOR,

    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  DEFAULT NOW()
);

-- Semantic search — IVFFlat for recall/speed tradeoff; HNSW for pure speed
CREATE INDEX idx_chunks_embedding_ivfflat
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200);

-- Full-text search
CREATE INDEX idx_chunks_fts      ON chunks USING GIN (fts_vector);

-- Metadata filters (used in WHERE before ANN — dramatically reduces scan)
CREATE INDEX idx_chunks_cik      ON chunks (cik);
CREATE INDEX idx_chunks_period   ON chunks (period_of_report, period_type);
CREATE INDEX idx_chunks_topics   ON chunks USING GIN (topics);
CREATE INDEX idx_chunks_flags    ON chunks (is_table, is_footnote, is_boilerplate);

-- Auto-update fts_vector on insert/update
CREATE OR REPLACE FUNCTION update_fts_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_fts
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_fts_vector();
