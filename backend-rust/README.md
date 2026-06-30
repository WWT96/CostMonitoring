# Cost Monitor Rust Backend POC

This is a non-production proof of concept for the future backend boundary. It is not wired into the Streamlit app yet.

## Commands

```powershell
cargo run -- health
cargo test
```

The current POC has no third-party Rust dependencies. The next slice should implement `db-health` and compare its JSON output against `scripts/ops_db_health.py`.
