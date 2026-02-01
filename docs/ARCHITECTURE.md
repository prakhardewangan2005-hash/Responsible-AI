# Architecture

## Components
- **Gradio UI**: collects inputs + renders outputs (cards + JSON panels + table)
- **Inference handler**: validation, rate-limit, request_id, timing
- **Model**: deterministic inference (loaded once)
- **Metrics store**: rolling window latency stats (p50/p95) + error-rate
- **Audit generator**: fairness audit JSON (group summaries)

## Data Flow
UI → validate/limit → predict → update metrics/audit → render outputs
