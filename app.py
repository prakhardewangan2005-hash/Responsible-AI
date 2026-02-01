import time
import uuid
import json
from collections import deque, defaultdict

import gradio as gr
import numpy as np

# -----------------------------
# Config
# -----------------------------
RATE_LIMIT_RPM = 30
WINDOW_SEC = 60
LAT_WINDOW = 300

# -----------------------------
# In-memory stores (demo-safe)
# -----------------------------
request_times = deque()
latencies = deque(maxlen=LAT_WINDOW)
recent_preds = deque(maxlen=10)

fairness_stats = defaultdict(lambda: {"count": 0, "gt50k": 0})

# -----------------------------
# Deterministic model (rule-based)
# -----------------------------
def deterministic_income_model(age, education, hours, sex):
    score = 0.0

    if age >= 40:
        score += 0.25
    if hours >= 40:
        score += 0.25
    if education in ["Masters", "Doctorate"]:
        score += 0.30
    elif education == "Bachelors":
        score += 0.15
    if sex == "male":
        score += 0.05

    score = min(score, 0.95)
    label = ">50K" if score >= 0.5 else "<=50K"
    return label, round(score, 2)

# -----------------------------
# Utilities
# -----------------------------
def rate_limit_check():
    now = time.time()
    while request_times and now - request_times[0] > WINDOW_SEC:
        request_times.popleft()
    if len(request_times) >= RATE_LIMIT_RPM:
        raise RuntimeError("Rate limit exceeded")
    request_times.append(now)

def latency_stats():
    if not latencies:
        return {"p50": 0, "p95": 0}
    arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }

# -----------------------------
# Predict handler
# -----------------------------
def predict(age, education, hours, sex, race):
    start = time.time()
    req_id = str(uuid.uuid4())[:8]

    rate_limit_check()

    if not (18 <= age <= 90):
        raise ValueError("Invalid age")
    if not (1 <= hours <= 80):
        raise ValueError("Invalid hours")

    label, conf = deterministic_income_model(age, education, hours, sex)

    latency = round((time.time() - start) * 1000, 2)
    latencies.append(latency)

    fairness_stats[race]["count"] += 1
    if label == ">50K":
        fairness_stats[race]["gt50k"] += 1

    recent_preds.appendleft({
        "time": int(time.time()),
        "label": label,
        "conf": conf,
        "lat_ms": latency,
        "edu": education,
        "sex": sex,
        "race": race,
        "age": age,
        "hours": hours
    })

    metrics = {
        "window": {
            "count": len(request_times),
            "max_window": RATE_LIMIT_RPM
        },
        "latency_ms": latency_stats(),
        "error_rate": 0.0,
        "limits": {
            "rate_limit_rpm": RATE_LIMIT_RPM,
            "recent_predictions": len(recent_preds)
        },
        "request_id": req_id
    }

    audit = {
        k: {
            "count": v["count"],
            "positive_rate": round(
                v["gt50k"] / v["count"], 2
            ) if v["count"] else 0
        }
        for k, v in fairness_stats.items()
    }

    return (
        label,
        conf,
        latency,
        json.dumps(metrics, indent=2),
        json.dumps(audit, indent=2),
        list(recent_preds)
    )

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="HF-RAI-Ops") as demo:
    gr.Markdown(
        """
# 🚀 HF-RAI-Ops — Responsible AI Income Prediction  
Educational demo only. Not for real-world decision making.
"""
    )

    with gr.Row():
        with gr.Column():
            age = gr.Slider(18, 90, value=30, label="Age")
            education = gr.Dropdown(
                ["HS-grad", "Bachelors", "Masters", "Doctorate"],
                value="Bachelors",
                label="Education"
            )
            hours = gr.Slider(1, 80, value=40, label="Hours per Week")
            sex = gr.Radio(["male", "female"], value="male", label="Sex")
            race = gr.Dropdown(
                ["white", "black", "asian-pac-islander", "other"],
                value="white",
                label="Race"
            )
            btn = gr.Button("Predict Income")

        with gr.Column():
            pred = gr.Textbox(label="Predicted Income")
            conf = gr.Textbox(label="Confidence")
            lat = gr.Textbox(label="Latency (ms)")
            metrics = gr.Code(label="Ops Metrics (JSON)")
            audit = gr.Code(label="Fairness Audit (JSON)")
            table = gr.Dataframe(
                headers=["time","label","conf","lat_ms","edu","sex","race","age","hours"],
                label="Recent Predictions (last 10)"
            )

    btn.click(
        predict,
        [age, education, hours, sex, race],
        [pred, conf, lat, metrics, audit, table]
    )

demo.launch()
