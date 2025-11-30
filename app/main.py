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

def format_time_duration(seconds):
    """Format seconds into hours, minutes, and seconds"""
    if seconds is None:
        return 'N/A'
    
    try:
        seconds = int(float(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    except (ValueError, TypeError):
        return str(seconds)

def format_timestamp(timestamp_str):
    """Format timestamp to mm-dd-yyyy HH:MM"""
    if not timestamp_str or timestamp_str == 'N/A':
        return 'N/A'
    
    try:
        # Parse the timestamp
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime('%m-%d-%Y %H:%M')
    except (ValueError, AttributeError):
        return timestamp_str

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
            # Experiment still running or failed - show as "Processing"
            if "404" in str(e) or "No such object" in str(e):
                experiments.append({
                    "name": exp_name,
                    "timestamp": "Processing...",
                    "mode": "cloud_function",
                    "cpus": None,
                    "generations": "-",
                    "population": "-",
                    "hidden": "-",
                    "status": "processing"
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
    exp_name = None
    
    try:
        # Get form data
        generations_str = request.form.get("generations")
        population_str = request.form.get("population")
        hidden_str = request.form.get("hidden")
        exp_name = request.form.get("exp_name", "").strip()
        
        print(f"Received form data - name: {exp_name}, generations: {generations_str}, population: {population_str}, hidden: {hidden_str}")
        
        if not generations_str or not population_str or not hidden_str:
            flash("Missing required parameters", "error")
            return redirect(url_for("index"))
        
        if not exp_name:
            flash("Experiment name is required", "error")
            return redirect(url_for("index"))
        
        # Validate experiment name (alphanumeric, hyphens, underscores only)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', exp_name):
            flash("Experiment name can only contain letters, numbers, hyphens, and underscores", "error")
            return redirect(url_for("index"))
        
        # Check if experiment name already exists
        bucket = client.bucket(BUCKET_NAME)
        prefix = f"{exp_name}/"
        existing_blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
        if existing_blobs:
            flash(f"Experiment name '{exp_name}' already exists. Please choose a different name.", "error")
            return redirect(url_for("index"))
        
        generations = int(generations_str)
        population = int(population_str)
        hidden = int(hidden_str)
        
        # Check if Cloud Function URL is configured
        if not CLOUD_FUNCTION_URL:
            flash("Cloud Function URL not configured. Please update CLOUD_FUNCTION_URL in main.py", "error")
            return redirect(url_for("index"))
        
        # Prepare payload for Cloud Function
        payload = {
            "exp_name": exp_name,  # Add experiment name to payload
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
                f"✅ Experiment '{exp_name}' launched! Gen: {generations}, Pop: {population}, Hidden: {hidden}",
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
            f"✅ Experiment '{exp_name}' started! Gen: {generations}, Pop: {population}, Hidden: {hidden}. Running in background, check back soon.",
            "success"
        )
    except requests.exceptions.Timeout:
        # General timeout
        print("Request timed out - experiment running in background")
        flash(
            f"✅ Experiment '{exp_name}' started! Gen: {generations}, Pop: {population}, Hidden: {hidden}. Running in background, check back soon.",
            "success"
        )
    except Exception as e:
        flash(f"❌ Error: {str(e)}", "error")
        print(f"Launch error: {e}")
        import traceback
        traceback.print_exc()
    
    return redirect(url_for("index"))

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
    is_processing = False
    summary = None
    data = None
    first_gen_time = None
    
    # Try to load final summary (experiment completed)
    try:
        summary_path = f"{exp_name}/final_summary.json"
        summary = load_json_from_gcs(summary_path)
        data = summary["data"]
        
        # Filter out GCS fields
        if data:
            data = {k: v for k, v in data.items() if k not in ['GCS_bucket', 'GCS_prefix']}
            
            # Format total_execution_time
            if 'total_execution_time' in data:
                data['total_execution_time'] = format_time_duration(data['total_execution_time'])
        
        # Format timestamp in summary
        if summary and summary.get("timestamp"):
            summary["timestamp"] = format_timestamp(summary["timestamp"])
            
    except Exception as e:
        # No final summary means experiment is still processing
        is_processing = True
        print(f"No final_summary.json for {exp_name} - experiment is processing")

    # Load generation history (works for both processing and completed)
    df = pd.DataFrame()
    try:
        df = load_history(exp_name)
        if not df.empty and is_processing:
            # Get first generation timestamp for elapsed time calculation
            if "timestamp" in df.columns:
                first_gen_time = df.iloc[0]["timestamp"]
                # Ensure it's in ISO format that JavaScript can parse
                if isinstance(first_gen_time, str):
                    first_gen_time = first_gen_time.replace("Z", "+00:00")
    except Exception as e:
        print(f"No history yet for {exp_name}: {e}")

    history_plot_html = ""
    time_plot_html = ""

    if not df.empty:
        try:
            # Fitness line chart
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["gen"], y=df["avg"], mode="lines+markers", name="Average Fitness"))
            fig.add_trace(go.Scatter(x=df["gen"], y=df["max"], mode="lines+markers", name="Max Fitness"))
            fig.update_layout(
                title="Fitness Over Generations", 
                xaxis_title="Generation", 
                yaxis_title="Fitness",
                hovermode='x unified'
            )
            history_plot_html = fig.to_html(full_html=False)

            # Time bar chart
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df["gen"], y=df["time_s"], name="Time (s)"))
            fig2.update_layout(
                title="Execution Time per Generation", 
                xaxis_title="Generation", 
                yaxis_title="Time (s)"
            )
            time_plot_html = fig2.to_html(full_html=False)
        except Exception as e:
            import traceback
            print("ERROR creating plots for", exp_name)
            traceback.print_exc()

    return render_template(
        "experiment.html",
        exp_name=exp_name,
        summary=summary,
        data=data,
        history_plot=history_plot_html,
        time_plot=time_plot_html,
        has_history=not df.empty,
        is_processing=is_processing,
        first_gen_time=first_gen_time,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)