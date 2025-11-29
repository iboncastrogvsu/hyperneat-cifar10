from flask import Flask, render_template, redirect, url_for, flash, request
from google.cloud import storage
from datetime import datetime
import json
import pandas as pd
import requests
import secrets

# ===== CONFIGURATION =====
BUCKET_NAME = "cis437-hyperneat-logs"
CLOUD_FUNCTION_URL = "https://hyperneat-evolve-800545748601.us-central1.run.app/evolve"

client = storage.Client()

# ===== FLASK APP =====
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Secure random secret key

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

def delete_experiment_from_gcs(exp_name):
    """Delete all files in the experiment folder from GCS"""
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{exp_name}/"))
    for blob in blobs:
        blob.delete()
    return len(blobs)

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

            mode = data.get("mode", "-")
            
            # Only include CPUs for local experiments
            cpus = data.get("cpus", "-") if mode in ["sequential", "ray"] else None

            experiments.append({
                "name": exp_name,
                "timestamp": timestamp_fmt,
                "mode": mode,
                "cpus": cpus,
                "generations": data.get("generations", "-"),
                "population": data.get("population", "-"),
                "hidden": data.get("hidden", "-"),
                "status": "completed"
            })
        except Exception as e:
            # Experiment still running or failed - show as "In Progress"
            if "404" in str(e) or "No such object" in str(e):
                experiments.append({
                    "name": exp_name,
                    "timestamp": "In Progress...",
                    "mode": "cloud_function",
                    "cpus": None,
                    "generations": "-",
                    "population": "-",
                    "hidden": "-",
                    "status": "running"
                })
            else:
                print(f"Error loading {exp_name}: {e}")

    return render_template("index.html", experiments=experiments)

@app.route("/launch", methods=["POST"])
def launch_experiment():
    """Launch a new experiment by calling the Cloud Function"""
    generations = None
    population = None
    hidden = None
    
    try:
        # Get form data
        generations_str = request.form.get("generations")
        population_str = request.form.get("population")
        hidden_str = request.form.get("hidden")
        
        print(f"Received form data - generations: {generations_str}, population: {population_str}, hidden: {hidden_str}")
        
        if not generations_str or not population_str or not hidden_str:
            flash("Missing required parameters", "error")
            return redirect(url_for("index"))
        
        generations = int(generations_str)
        population = int(population_str)
        hidden = int(hidden_str)
        
        # Check if Cloud Function URL is configured
        if not CLOUD_FUNCTION_URL or CLOUD_FUNCTION_URL == "YOUR_CLOUD_FUNCTION_URL_HERE":
            flash("Cloud Function URL not configured. Please update CLOUD_FUNCTION_URL in main.py", "error")
            return redirect(url_for("index"))
        
        # Prepare payload for Cloud Function
        payload = {
            "generations": generations,
            "population": population,
            "hidden": hidden,
            "seed": 123,
            "subset_size": 2000
        }
        
        # Call the Cloud Function
        print(f"Launching experiment with payload: {payload}")
        print(f"Calling Cloud Function at: {CLOUD_FUNCTION_URL}")
        
        # Make async call to Cloud Function
        response = requests.post(
            CLOUD_FUNCTION_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        print(f"Cloud Function response status: {response.status_code}")
        print(f"Cloud Function response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            flash(
                f"✅ Experiment launched! Gen: {generations}, Pop: {population}, Hidden: {hidden}",
                "success"
            )
        else:
            # Show error details
            try:
                error_data = response.json()
                error_msg = error_data.get("error", response.text)
            except:
                error_msg = response.text
            
            flash(f"❌ Launch failed: {error_msg}", "error")
            
    except requests.exceptions.ReadTimeout:
        # Timeout is expected - experiment is running in background
        print("Request timed out - experiment running in background")
        flash(
            f"✅ Experiment started! Gen: {generations}, Pop: {population}, Hidden: {hidden}. Running in background, check back soon.",
            "success"
        )
    except requests.exceptions.Timeout:
        # General timeout
        print("Request timed out - experiment running in background")
        flash(
            f"✅ Experiment started! Gen: {generations}, Pop: {population}, Hidden: {hidden}. Running in background, check back soon.",
            "success"
        )
    except Exception as e:
        flash(f"❌ Error: {str(e)}", "error")
        print(f"Launch error: {e}")
        import traceback
        traceback.print_exc()
    
    return redirect(url_for("index"))

@app.route("/logs")
def view_logs():
    """View recent Cloud Function logs (optional feature)"""
    try:
        from google.cloud import logging as cloud_logging
        
        logging_client = cloud_logging.Client()
        logger = logging_client.logger("cloudfunctions.googleapis.com%2Fcloud-functions")
        
        # Get last 50 log entries
        entries = list(logger.list_entries(max_results=50, order_by=cloud_logging.DESCENDING))
        
        logs = []
        for entry in entries:
            logs.append({
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
                "severity": entry.severity,
                "message": str(entry.payload)
            })
        
        return render_template("logs.html", logs=logs)
    except Exception as e:
        return f"<h3>Error loading logs: {e}</h3><p>Make sure google-cloud-logging is installed</p>", 500

@app.route("/local-experiments")
def local_experiments():
    """View local experiments results"""
    return render_template("local_experiments.html")

@app.route("/delete/<exp_name>", methods=["POST"])
def delete_experiment(exp_name):
    """Delete an experiment from GCS and redirect back to index"""
    try:
        flash(f"Successfully deleted experiment '{exp_name}'", "success")
    except Exception as e:
        flash(f"Error deleting experiment '{exp_name}': {str(e)}", "error")
    return redirect(url_for("index"))

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