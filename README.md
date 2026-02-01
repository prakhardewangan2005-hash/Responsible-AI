# 🚀 HF-RAI-Ops — Responsible AI Income Prediction

**Live Demo:** https://huggingface.co/spaces/prakhardewangan/hf-rai-ops 

A production-style ML micro-demo that showcases:
- deterministic inference pipeline (no flaky retraining)
- p50/p95 latency tracking + error-rate
- rate limiting + input validation
- structured logs with request_id
- fairness audit + model-card style documentation

> ⚠️ Educational demo only. Not for real-world decision making.

---

## What it does
1) User inputs (age, education, hours/week, sex, race)  
2) Model predicts income bucket (<=50K / >50K) with confidence  
3) System outputs:
- **Predicted label**
- **Confidence**
- **Latency**
- **Ops metrics JSON**
- **Fairness audit JSON**
- **Recent predictions table**

---

## Key Engineering Highlights
- **Deterministic model**: stable outputs, avoids “Train” button randomness
- **Reliability**: input validation, safe defaults, guarded preprocessing
- **Observability**: request_id, structured logs, p50/p95 latency window
- **Abuse protection**: rate limit (RPM) + bounded history
- **Responsible AI**: group-wise outcome summary + audit JSON

---

## Architecture 
Client (Gradio UI)
   → Inference Handler (validation + rate limit)
      → Model (deterministic, loaded once)
      → Metrics Store (in-memory rolling window)
      → Audit Generator (group summaries)
   → UI renders cards + JSON panels + table

---

## Local Run
```bash
pip install -r requirements.txt
python app.py
