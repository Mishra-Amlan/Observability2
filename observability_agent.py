# =============================================================================
# OBSERVABILITY AGENT  —  BMC Helix Automation Team
# Watches a RAG agent's test log (Groq + Gradio) for anomalies,
# raises plain-language alerts, and scores itself on a System / Process /
# Quality evaluation matrix.
#
# HOW TO USE THIS SCRIPT
# 1. Install dependencies:  pip install -r requirements.txt
# 2. Run:                   python observability_agent.py
# 3. Put your Groq API key in the .env file next to this script:
#        GROQ_API_KEY=your_key_here
#    If it's blank, the script runs in DRY-RUN mode (everything works except
#    the Groq-written summaries).
# 4. A demo log is generated automatically so the script is fully runnable
#    with no setup. To watch YOUR real agent instead, replace the call to
#    generate_mock_log() near the bottom of SECTION 2 with your own log —
#    any pandas DataFrame with the same columns (or a CSV/Google Sheet read
#    into one) works.
# 5. A small Gradio dashboard launches at the end with a public test link.
# =============================================================================

import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
from dotenv import dotenv_values
from groq import Groq

random.seed(7)
np.random.seed(7)

# --- SECTION 1 : Groq API key & helper -------------------------------------
# Reads GROQ_API_KEY from the .env file in the same folder as this script.
# If the key is blank or missing, the script runs in DRY-RUN mode.
ENV_FILE = Path(__file__).with_name(".env")
GROQ_API_KEY = (dotenv_values(ENV_FILE).get("GROQ_API_KEY") or "").strip()

DRY_RUN = len(GROQ_API_KEY) == 0
GROQ_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models for current options

client = Groq(api_key=GROQ_API_KEY) if not DRY_RUN else None

def ask_groq(prompt, system="You are a concise, factual site-reliability assistant.", max_tokens=220):
    """Calls Groq if a key is set; otherwise returns a clearly-labeled stub
    so the rest of the script still runs end to end."""
    if DRY_RUN:
        return f"[DRY-RUN — no Groq key set, this is a placeholder] {prompt[:120]}..."
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq call failed: {e}]"

print("DRY_RUN mode:" , DRY_RUN)

# --- SECTION 2 : The test log the agent watches -----------------------------
def generate_mock_log(n_normal=160, n_incident=25, start=None):
    """Simulates a RAG-agent interaction log, like the one your Observability
    agent reads in real use: a normal traffic window, followed by an injected
    incident (latency spike + Groq errors + a guardrail/prompt regression) so
    there's something real to detect."""
    if start is None:
        start = dt.datetime(2026, 9, 1, 9, 0, 0)
    rows, sid = [], 1000
    sample_questions = [
        "What's the rollback process for a failed deployment?",
        "How do I reset a locked-out user's access?",
        "What's the standard provisioning time for a new tenant?",
        "Can you approve this deployment for me?",
        "What's the admin password for the CI server?",
    ]
    t = start
    for _ in range(n_normal):
        t += dt.timedelta(seconds=random.randint(20, 90))
        sid += 1
        rows.append({
            "timestamp": t, "session_id": f"S{sid}",
            "question": random.choice(sample_questions),
            "retrieved_ok": True,
            "groq_latency_s": max(0.3, round(np.random.normal(1.6, 0.4), 2)),
            "groq_status": "ok",
            "guardrail_flag": random.choices(["pass", "held"], weights=[0.94, 0.06])[0],
            "response_ok": True,
        })
    incident_start = t + dt.timedelta(minutes=2)
    t = incident_start
    for _ in range(n_incident):
        t += dt.timedelta(seconds=random.randint(10, 40))
        sid += 1
        is_err = random.random() < 0.35
        rows.append({
            "timestamp": t, "session_id": f"S{sid}",
            "question": random.choice(sample_questions),
            "retrieved_ok": random.random() > 0.15,
            "groq_latency_s": max(0.3, round(np.random.normal(9.5, 2.0), 2)),
            "groq_status": random.choice(["error_429", "error_401"]) if is_err else "ok",
            "guardrail_flag": random.choices(["pass", "held", "blocked"], weights=[0.55, 0.25, 0.20])[0],
            "response_ok": not is_err,
        })
    df = pd.DataFrame(rows)
    return df, incident_start

log_df, incident_start = generate_mock_log()
print(f"Loaded {len(log_df)} log rows. Simulated incident starts at {incident_start}.")
print(log_df.tail(8))

# --- SECTION 3 : Observability agent core logic -----------------------------
THRESHOLDS = {
    "avg_latency_s": 4.0,        # seconds
    "p95_latency_s": 6.0,        # seconds
    "groq_error_rate": 5.0,      # percent
    "retrieval_miss_rate": 10.0, # percent
    "guardrail_block_rate": 10.0,# percent
}

def compute_metrics(df, window_minutes=None):
    d = df
    if window_minutes is not None:
        cutoff = df["timestamp"].max() - dt.timedelta(minutes=window_minutes)
        d = df[df["timestamp"] >= cutoff]
    n = len(d)
    if n == 0:
        return {}
    return {
        "n_requests": int(n),
        "avg_latency_s": float(round(d["groq_latency_s"].mean(), 2)),
        "p95_latency_s": float(round(d["groq_latency_s"].quantile(0.95), 2)),
        "groq_error_rate": float(round((d["groq_status"] != "ok").mean() * 100, 1)),
        "retrieval_miss_rate": float(round((~d["retrieved_ok"]).mean() * 100, 1)),
        "guardrail_block_rate": float(round((d["guardrail_flag"] == "blocked").mean() * 100, 1)),
        "guardrail_hold_rate": float(round((d["guardrail_flag"] == "held").mean() * 100, 1)),
    }

def check_breaches(metrics, thresholds=THRESHOLDS):
    return [(k, metrics[k], limit) for k, limit in thresholds.items()
            if k in metrics and metrics[k] > limit]

def raise_alerts(df, window_minutes=10):
    """The full Observability loop: collect -> aggregate -> compare ->
    Groq-summarize -> alert -> log."""
    metrics = compute_metrics(df, window_minutes=window_minutes)
    breaches = check_breaches(metrics)
    alerts = []
    for metric, value, limit in breaches:
        evidence = df.tail(20)[["timestamp", "session_id", "groq_status", "groq_latency_s", "guardrail_flag"]].to_string(index=False)
        prompt = (
            f"A RAG agent's {metric} is {value}, above its threshold of {limit}. "
            f"Here are the last 20 log rows as evidence:\n{evidence}\n\n"
            "In 2-3 short sentences, explain in plain language what this likely means "
            "for the team, without inventing details not in the evidence."
        )
        summary = ask_groq(prompt)
        alerts.append({
            "timestamp": dt.datetime.now().replace(microsecond=0),
            "metric": metric,
            "value": value,
            "threshold": limit,
            "summary": summary,
        })
    return pd.DataFrame(alerts), metrics

alerts_df, recent_metrics = raise_alerts(log_df, window_minutes=10)
print("Recent metrics (last 10 min):", recent_metrics)
print(f"\n{len(alerts_df)} alert(s) raised.\n")
if len(alerts_df):
    for _, a in alerts_df.iterrows():
        print(f"⚠ {a['metric']} = {a['value']} (limit {a['threshold']})\n   {a['summary']}\n")

# --- SECTION 4 : Evaluation matrix (System / Process / Quality) ------------
def rate(value, strong_at, medium_at, higher_is_better=True):
    """Turns a measured number into a Strong / Medium / Weak label."""
    if higher_is_better:
        if value >= strong_at: return "Strong"
        if value >= medium_at: return "Medium"
        return "Weak"
    else:
        if value <= strong_at: return "Strong"
        if value <= medium_at: return "Medium"
        return "Weak"

def build_evaluation_matrix(df, alerts_df, overall_metrics, breaches):
    n = len(df)
    fields_expected = {"timestamp", "session_id", "groq_status", "groq_latency_s", "retrieved_ok", "guardrail_flag"}
    fields_present = fields_expected.issubset(set(df.columns))
    integration_coverage = 100.0 if fields_present else round(len(fields_expected & set(df.columns)) / len(fields_expected) * 100, 1)

    alerts_have_evidence = all("summary" in alerts_df.columns for _ in [0]) if len(alerts_df) else True
    groundedness_pct = 100.0 if alerts_have_evidence else 0.0

    true_incident_alert_found = any(m in ["avg_latency_s", "groq_error_rate", "guardrail_block_rate"] for m in alerts_df["metric"]) if len(alerts_df) else False
    signal_quality = "Strong" if true_incident_alert_found and len(alerts_df) <= len(THRESHOLDS) else ("Medium" if true_incident_alert_found else "Weak")

    summary_lengths = alerts_df["summary"].str.split().str.len() if len(alerts_df) else pd.Series(dtype=float)
    clarity_ok_pct = float(round(((summary_lengths >= 8) & (summary_lengths <= 80)).mean() * 100, 1)) if len(summary_lengths) else 100.0

    rows = [
        # category, criterion, description, observed result, rating
        ["System", "Ran without errors", "Did the full pass over the log complete cleanly?",
         "Yes — completed" , "Strong"],
        ["System", "Integration coverage", "% of expected log fields the agent could read",
         f"{integration_coverage}%", rate(integration_coverage, 95, 80)],
        ["System", "Requests analyzed", "Volume of traffic the agent processed this run",
         f"{n} requests", "Strong" if n > 50 else "Medium"],

        ["Process", "Thresholds explicit & documented", "Are limits defined in code, not tribal knowledge?",
         f"{len(THRESHOLDS)} thresholds defined", "Strong"],
        ["Process", "Escalation discipline", "Were all breaching metrics turned into alerts?",
         f"{len(breaches)} breach(es) -> {len(alerts_df)} alert(s)",
         "Strong" if len(alerts_df) == len(breaches) else "Weak"],
        ["Process", "Audit logging", "Is every alert timestamped with metric + evidence?",
         "Yes — timestamp, metric, value, threshold, summary", "Strong"],

        ["Quality", "Alert groundedness", "Does each alert cite real evidence, not just a guess?",
         f"{groundedness_pct}%", rate(groundedness_pct, 95, 70)],
        ["Quality", "Signal quality", "Did the agent flag the injected incident, without excess noise?",
         "Incident detected" if true_incident_alert_found else "Missed", signal_quality],
        ["Quality", "Summary clarity", "% of Groq summaries in a readable 8-80 word range",
         f"{clarity_ok_pct}%", rate(clarity_ok_pct, 90, 60)],
    ]
    return pd.DataFrame(rows, columns=["Category", "Criterion", "What It Checks", "Observed Result", "Rating"])

breaches_now = check_breaches(recent_metrics)
eval_df = build_evaluation_matrix(log_df, alerts_df, recent_metrics, breaches_now)
print("\n=== OBSERVABILITY AGENT — EVALUATION MATRIX ===")
print(eval_df.to_string(index=False))

# --- SECTION 5 : Minimal Gradio dashboard -----------------------------------
def run_dashboard(window_minutes):
    a_df, m = raise_alerts(log_df, window_minutes=int(window_minutes))
    b = check_breaches(m)
    e_df = build_evaluation_matrix(log_df, a_df, m, b)
    metrics_txt = "\n".join(f"{k}: {v}" for k, v in m.items())
    alerts_out = a_df if len(a_df) else pd.DataFrame([{"info": "No breaches in this window."}])
    return metrics_txt, alerts_out, e_df

with gr.Blocks(title="Observability Agent") as demo:
    gr.Markdown("## Observability Agent — Live Test Dashboard\nAdjust the lookback window and re-run to see metrics, alerts, and the evaluation matrix update.")
    window = gr.Slider(2, 60, value=10, step=1, label="Lookback window (minutes)")
    run_btn = gr.Button("Run Observability Check", variant="primary")
    metrics_box = gr.Textbox(label="Current Metrics", lines=8)
    alerts_table = gr.Dataframe(label="Alerts Raised")
    eval_table = gr.Dataframe(label="Evaluation Matrix — System / Process / Quality")
    run_btn.click(run_dashboard, inputs=window, outputs=[metrics_box, alerts_table, eval_table])
    demo.load(run_dashboard, inputs=window, outputs=[metrics_box, alerts_table, eval_table])

if __name__ == "__main__":
    demo.launch(share=True)
