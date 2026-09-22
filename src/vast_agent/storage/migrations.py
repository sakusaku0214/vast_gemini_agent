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
)
