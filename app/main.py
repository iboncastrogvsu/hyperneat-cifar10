from flask import Flask, render_template
from google.cloud import storage
from datetime import datetime
import json
import pandas as pd

# ===== CONFIGURATION =====
BUCKET_NAME = "cis437-hyperneat-logs"

client = storage.Client()

# ===== FLASK APP =====
app = Flask(__name__)

def list_experiments():
    """List all experiment folders containing final_summary.json"""
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs())
    experiments = []
    for blob in blobs:
        if blob.name.endswith("final_summary.json"):
            exp_name = blob.name.split("/")[0]
            experiments.append(exp_name)
    return sorted(set(experiments))

def load_json_from_gcs(path):
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    return json.loads(blob.download_as_text())

def load_history(exp_name):
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"{exp_name}/history/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    records = []
    for blob in sorted(blobs, key=lambda x: x.name):
        if blob.name.endswith(".json"):
            data = json.loads(blob.download_as_text())
            records.append(data)
    return pd.DataFrame(records) if records else pd.DataFrame()

@app.route("/")
def index():
    experiments = []
    # List all blobs in the bucket
    blobs = list(client.list_blobs(BUCKET_NAME))

    exp_folders = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 1:
            exp_folders.add(parts[0])

    for exp_name in sorted(exp_folders):
        try:
            summary = load_json_from_gcs(f"{exp_name}/final_summary.json")
            data = summary.get("data", summary)

            # Format timestamp nicely
            timestamp = summary.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_fmt = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_fmt = timestamp

            experiments.append({
                "name": exp_name,
                "timestamp": timestamp_fmt,
                "mode": data.get("mode", "-"),
                "cpus": data.get("cpus", "-"),
                "generations": data.get("generations", "-"),
                "population": data.get("population", "-"),
                "hidden": data.get("hidden", "-"),
            })
        except Exception as e:
            print(f"Skipping {exp_name}: {e}")

    return render_template("index.html", experiments=experiments)



@app.route("/experiment/<exp_name>")
def experiment(exp_name):
    try:
        summary_path = f"{exp_name}/final_summary.json"
        summary = load_json_from_gcs(summary_path)
        data = summary["data"]
    except Exception as e:
        import traceback
        print("ERROR loading final_summary.json for", exp_name)
        traceback.print_exc()
        return f"<h3>Error loading summary: {e}</h3>", 500

    try:
        df = load_history(exp_name)
    except Exception as e:
        import traceback
        print("ERROR loading history for", exp_name)
        traceback.print_exc()
        return f"<h3>Error loading history: {e}</h3>", 500

    history_plot_html = ""
    time_plot_html = ""

    if not df.empty:
        try:
            # Fitness line chart
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["gen"], y=df["avg"], mode="lines", name="Average Fitness"))
            fig.add_trace(go.Scatter(x=df["gen"], y=df["max"], mode="lines", name="Max Fitness"))
            fig.update_layout(title="Fitness Over Generations", xaxis_title="Generation", yaxis_title="Fitness")
            history_plot_html = fig.to_html(full_html=False)

            # Time bar chart
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df["gen"], y=df["time_s"], name="Time (s)"))
            fig2.update_layout(title="Execution Time per Generation", xaxis_title="Generation", yaxis_title="Time (s)")
            time_plot_html = fig2.to_html(full_html=False)
        except Exception as e:
            import traceback
            print("ERROR creating plots for", exp_name)
            traceback.print_exc()
            return f"<h3>Error creating plots: {e}</h3>", 500

    return render_template(
        "experiment.html",
        exp_name=exp_name,
        summary=summary,
        data=data,
        history_plot=history_plot_html,
        time_plot=time_plot_html,
        has_history=not df.empty,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
