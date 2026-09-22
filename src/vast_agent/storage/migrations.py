MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE schema_version (version INTEGER NOT NULL);
    INSERT INTO schema_version VALUES (1);
    CREATE TABLE hosts (
      name TEXT PRIMARY KEY, vast_id INTEGER, address TEXT NOT NULL,
      ssh_user TEXT NOT NULL, ssh_port INTEGER NOT NULL, enabled INTEGER NOT NULL
    );
    CREATE TABLE observations (
      id INTEGER PRIMARY KEY, host TEXT NOT NULL, observed_at TEXT NOT NULL,
      payload_json TEXT NOT NULL, raw_log_path TEXT,
      FOREIGN KEY(host) REFERENCES hosts(name)
    );
    CREATE TABLE tool_runs (
      id INTEGER PRIMARY KEY, observation_id INTEGER, tool_name TEXT NOT NULL,
      success INTEGER NOT NULL, exit_code INTEGER, duration_ms INTEGER NOT NULL,
      error_code TEXT, raw_log_path TEXT,
      FOREIGN KEY(observation_id) REFERENCES observations(id)
    );
    CREATE TABLE incidents (
      id INTEGER PRIMARY KEY, host TEXT NOT NULL, primary_signature TEXT NOT NULL,
      opened_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, status TEXT NOT NULL,
      severity TEXT NOT NULL, summary TEXT NOT NULL, dedup_key TEXT NOT NULL,
      UNIQUE(dedup_key, status)
    );
    CREATE INDEX incidents_host_signature ON incidents(host, primary_signature);
    """,
    """
    CREATE TABLE token_usage (
      id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, purpose TEXT NOT NULL,
      model TEXT NOT NULL, thinking_level TEXT NOT NULL,
      input_tokens INTEGER, output_tokens INTEGER, thought_tokens INTEGER,
      cached_tokens INTEGER, tool_use_tokens INTEGER, total_tokens INTEGER
    );
    CREATE INDEX token_usage_created_at ON token_usage(created_at);
    """,
    """
    CREATE TABLE jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, host TEXT,
      status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT,
      finished_at TEXT, request_summary TEXT NOT NULL, result_summary TEXT,
      error_code TEXT
    );
    CREATE INDEX jobs_created_at ON jobs(created_at);
    CREATE TABLE conversation_state (
      owner_id TEXT NOT NULL, channel_id TEXT NOT NULL, last_host TEXT,
      last_job_id INTEGER, last_scope TEXT, updated_at TEXT NOT NULL,
      PRIMARY KEY(owner_id, channel_id)
    );
    """,
    """
    CREATE TABLE action_proposals (
      id INTEGER PRIMARY KEY AUTOINCREMENT, host TEXT NOT NULL, action_type TEXT NOT NULL,
      params_json TEXT NOT NULL, risk_class TEXT NOT NULL, status TEXT NOT NULL,
      created_at TEXT NOT NULL, expires_at TEXT NOT NULL, created_by TEXT NOT NULL,
      preflight_json TEXT NOT NULL, preflight_fingerprint TEXT NOT NULL, job_id INTEGER,
      approved_at TEXT, approved_by TEXT, consumed_at TEXT, invalidated_reason TEXT
    );
    CREATE INDEX action_proposals_status_expiry ON action_proposals(status, expires_at);
    CREATE TABLE action_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT, proposal_id INTEGER NOT NULL UNIQUE, job_id INTEGER,
      started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, exit_code INTEGER,
      error_code TEXT, verify_summary TEXT, point_of_no_return INTEGER NOT NULL DEFAULT 0,
      FOREIGN KEY(proposal_id) REFERENCES action_proposals(id)
    );
    CREATE TABLE action_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT, proposal_id INTEGER NOT NULL, run_id INTEGER,
      created_at TEXT NOT NULL, event TEXT NOT NULL, detail TEXT,
      FOREIGN KEY(proposal_id) REFERENCES action_proposals(id),
      FOREIGN KEY(run_id) REFERENCES action_runs(id)
    );
    """,
)
