use std::env;

const CONTRACT_VERSION: &str = "0.1.0";
const SERVICE_NAME: &str = "cost-monitor-backend";

fn health() -> String {
    format!(
        "{{\"ok\":true,\"service\":\"{}\",\"contract_version\":\"{}\"}}",
        SERVICE_NAME, CONTRACT_VERSION
    )
}

fn usage() -> String {
    format!(
        "{{\"ok\":false,\"service\":\"{}\",\"error\":\"supported commands: health\"}}",
        SERVICE_NAME
    )
}

fn main() {
    let command = env::args().nth(1).unwrap_or_else(|| "health".to_string());
    let payload = match command.as_str() {
        "health" => health(),
        _ => usage(),
    };
    println!("{}", payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_reports_contract_version() {
        let payload = health();
        assert!(payload.contains("\"ok\":true"));
        assert!(payload.contains(SERVICE_NAME));
        assert!(payload.contains(CONTRACT_VERSION));
    }
}
