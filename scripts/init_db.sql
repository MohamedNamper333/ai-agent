-- AI Agent Database Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users
CREATE TABLE IF NOT EXISTS users (
    user_id     VARCHAR(64) PRIMARY KEY,
    username    VARCHAR(255) UNIQUE NOT NULL,
    role        VARCHAR(32) NOT NULL DEFAULT 'basic',
    api_key     VARCHAR(128) UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login  TIMESTAMPTZ,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    conv_id     VARCHAR(64) PRIMARY KEY,
    user_id     VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    title       VARCHAR(255),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    conv_id     VARCHAR(64) REFERENCES conversations(conv_id) ON DELETE CASCADE,
    role        VARCHAR(32) NOT NULL,
    content     TEXT NOT NULL,
    token_est   INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, created_at);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(64),
    action      VARCHAR(128) NOT NULL,
    resource    VARCHAR(255),
    ip_address  INET,
    user_agent  TEXT,
    request_id  UUID DEFAULT uuid_generate_v4(),
    status      VARCHAR(32),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action, created_at);

-- Tool execution history
CREATE TABLE IF NOT EXISTS tool_executions (
    id              BIGSERIAL PRIMARY KEY,
    conv_id         VARCHAR(64),
    tool_name       VARCHAR(128) NOT NULL,
    arguments       JSONB,
    result          TEXT,
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    execution_ms    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tool_exec_conv ON tool_executions(conv_id);
CREATE INDEX IF NOT EXISTS idx_tool_exec_name ON tool_executions(tool_name);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    notif_id    VARCHAR(32) PRIMARY KEY,
    user_id     VARCHAR(64) NOT NULL,
    title       VARCHAR(255) NOT NULL,
    message     TEXT NOT NULL,
    ntype       VARCHAR(32) DEFAULT 'info',
    is_read     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, is_read);

-- Learning data (Deep Learning feedback)
CREATE TABLE IF NOT EXISTS learning_feedback (
    id          BIGSERIAL PRIMARY KEY,
    conv_id     VARCHAR(64),
    message_id  BIGINT REFERENCES messages(id),
    feedback    VARCHAR(16),  -- 'positive', 'negative', 'neutral'
    tool_used   VARCHAR(128),
    embedding   BYTEA,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
