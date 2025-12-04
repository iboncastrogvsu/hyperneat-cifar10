from flask import Flask, render_template, redirect, url_for, flash, request
from google.cloud import storage
from kubernetes import client, config
from datetime import datetime
import json
import pandas as pd
import requests
import secrets

# ===== CONFIGURATION =====
BUCKET_NAME = "cis437-hyperneat-logs"
CLOUD_FUNCTION_URL = "https://hyperneat-evolve-800545748601.us-central1.run.app/evolve"
K8S_CLUSTER_NAME = "hyperneat"
K8S_NAMESPACE = "default"

storage_client = storage.Client()

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()

k8s_batch_v1 = client.BatchV1Api()
k8s_custom_api = client.CustomObjectsApi()

# ===== FLASK APP =====
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

def list_experiments():
    """List all experiment folders containing final_summary.json"""
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs())
    experiments = []
    for blob in blobs:
        if blob.name.endswith("final_summary.json"):
            exp_name = blob.name.split("/")[0]
            experiments.append(exp_name)
    return sorted(set(experiments))

def load_json_from_gcs(path):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    return json.loads(blob.download_as_text())

def load_history(exp_name):
    bucket = storage_client.bucket(BUCKET_NAME)
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
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{exp_name}/"))
    for blob in blobs:
        blob.delete()
    return len(blobs)

def create_rayjob_yaml_with_secret(exp_name, generations, population, hidden):
    """Create a RayJob YAML configuration with GCS secret mounted"""
    rayjob = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": f"hyperneat-{exp_name}",
            "namespace": K8S_NAMESPACE
        },
        "spec": {
            "entrypoint": f"python main.py --mode=ray --name_exp={exp_name} --generations={generations} --population={population} --hidden={hidden}",
            "submitterPodTemplate": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "ray-job-submitter",
                        "image": "gcr.io/term-project-ibon-castro/hyperneat:latest",
                        "imagePullPolicy": "Always",
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "1Gi"},
                            "limits": {"cpu": "2000m", "memory": "3Gi"}
                        },
                        "env": [
                            {
                                "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                "value": "/var/secrets/google/key.json"
                            },
                            {
                                "name": "COMPUTE_PLATFORM",
                                "value": "kubernetes"
                            }
                        ],
                        "volumeMounts": [{
                            "name": "gcs-key",
                            "mountPath": "/var/secrets/google",
                            "readOnly": True
                        }]
                    }],
                    "volumes": [{
                        "name": "gcs-key",
                        "secret": {
                            "secretName": "gcs-key"
                        }
                    }]
                }
            },
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 300,
            "rayClusterSpec": {
                "rayVersion": "2.9.0",
                "headGroupSpec": {
                    "serviceType": "ClusterIP",
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "num-cpus": "2"
                    },
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "ray-head",
                                "image": "gcr.io/term-project-ibon-castro/hyperneat:latest",
                                "imagePullPolicy": "Always",
                                "resources": {
                                    "requests": {"cpu": "2000m", "memory": "4Gi"},
                                    "limits": {"cpu": "4000m", "memory": "8Gi"}
                                },
                                "ports": [
                                    {"containerPort": 6379, "name": "gcs"},
                                    {"containerPort": 8265, "name": "dashboard"},
                                    {"containerPort": 10001, "name": "client"}
                                ],
                                "env": [
                                    {
                                        "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                        "value": "/var/secrets/google/key.json"
                                    },
                                    {
                                        "name": "COMPUTE_PLATFORM",
                                        "value": "kubernetes"
                                    }
                                ],
                                "volumeMounts": [
                                    {"name": "data-dir", "mountPath": "/home/ray/data"},
                                    {"name": "shared-mem", "mountPath": "/dev/shm"},
                                    {"name": "gcs-key", "mountPath": "/var/secrets/google", "readOnly": True}
                                ]
                            }],
                            "volumes": [
                                {"name": "data-dir", "emptyDir": {}},
                                {"name": "shared-mem", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
                                {"name": "gcs-key", "secret": {"secretName": "gcs-key"}}
                            ]
                        }
                    }
                },
                "workerGroupSpecs": [{
                    "replicas": 2,
                    "minReplicas": 2,
                    "maxReplicas": 4,
                    "groupName": "workers",
                    "rayStartParams": {"num-cpus": "4"},
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "ray-worker",
                                "image": "gcr.io/term-project-ibon-castro/hyperneat:latest",
                                "imagePullPolicy": "Always",
                                "resources": {
                                    "requests": {"cpu": "2000m", "memory": "4Gi"},
                                    "limits": {"cpu": "4000m", "memory": "8Gi"}
                                },
                                "env": [
                                    {
                                        "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                        "value": "/var/secrets/google/key.json"
                                    },
                                    {
                                        "name": "COMPUTE_PLATFORM",
                                        "value": "kubernetes"
                                    }
                                ],
                                "volumeMounts": [
                                    {"name": "shared-mem", "mountPath": "/dev/shm"},
                                    {"name": "data-dir", "mountPath": "/home/ray/data"},
                                    {"name": "gcs-key", "mountPath": "/var/secrets/google", "readOnly": True}
                                ]
                            }],
                            "volumes": [
                                {"name": "shared-mem", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}},
                                {"name": "data-dir", "emptyDir": {}},
                                {"name": "gcs-key", "secret": {"secretName": "gcs-key"}}
                            ]
                        }
                    }
                }]
            }
        }
    }
    return rayjob

def launch_kubernetes_job(exp_name, generations, population, hidden):
    """Launch a RayJob on Kubernetes cluster"""
    try:
        rayjob = create_rayjob_yaml_with_secret(exp_name, generations, population, hidden)
        
        k8s_custom_api.create_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=K8S_NAMESPACE,
            plural="rayjobs",
            body=rayjob
        )
        
        return True, "RayJob created successfully"
    except Exception as e:
        return False, str(e)

def delete_rayjob(exp_name):
    """Delete a RayJob from Kubernetes"""
    try:
        k8s_custom_api.delete_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=K8S_NAMESPACE,
            plural="rayjobs",
            name=f"hyperneat-{exp_name}"
        )
        return True
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return True
        return False

def get_rayjob_status(exp_name):
    """Get the status of a RayJob"""
    try:
        job = k8s_custom_api.get_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=K8S_NAMESPACE,
            plural="rayjobs",
            name=f"hyperneat-{exp_name}"
        )
        return job.get("status", {})
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return None
        raise

def format_time_duration(seconds):
    """Format seconds into hours, minutes, and seconds"""
    if seconds is None:
        return 'N/A'
    
    try:
        seconds = float(seconds)
        
        if seconds < 1:
            milliseconds = int(seconds * 1000)
            return f"{milliseconds}ms"
        
        seconds = int(seconds)
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
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime('%m-%d-%Y %H:%M')
    except (ValueError, AttributeError):
        return timestamp_str

@app.route("/")
def index():
    experiments = []
    blobs = list(storage_client.list_blobs(BUCKET_NAME))

    exp_folders = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 1:
            exp_folders.add(parts[0])

    for exp_name in sorted(exp_folders):
        try:
            summary = load_json_from_gcs(f"{exp_name}/final_summary.json")
            data = summary.get("data", summary)

            timestamp = summary.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_fmt = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_fmt = timestamp


            mode = data.get("mode", "-")
            
            compute_platform = data.get("compute_platform", None)
            
            if compute_platform == "kubernetes":
                mode = "kubernetes"
            
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
            if "404" in str(e) or "No such object" in str(e):
                k8s_status = get_rayjob_status(exp_name)
                if k8s_status:
                    experiments.append({
                        "name": exp_name,
                        "timestamp": "Processing...",
                        "mode": "kubernetes",
                        "cpus": None,
                        "generations": "-",
                        "population": "-",
                        "hidden": "-",
                        "status": "processing"
                    })
                else:
                    experiments.append({
                        "name": exp_name,
                        "timestamp": "Processing...",
                        "mode": "cloud_run",
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
    """Launch a new experiment by calling Cloud Run or Kubernetes"""
    generations = None
    population = None
    hidden = None
    exp_name = None
    
    try:
        generations_str = request.form.get("generations")
        population_str = request.form.get("population")
        hidden_str = request.form.get("hidden")
        exp_name = request.form.get("exp_name", "").strip()
        compute_option = request.form.get("compute_option", "cloud_run")
        
        print(f"Received form data - name: {exp_name}, generations: {generations_str}, population: {population_str}, hidden: {hidden_str}, compute: {compute_option}")
        
        if not generations_str or not population_str or not hidden_str:
            flash("Missing required parameters", "error")
            return redirect(url_for("index"))
        
        if not exp_name:
            flash("Experiment name is required", "error")
            return redirect(url_for("index"))
        
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', exp_name):
            flash("Experiment name can only contain letters, numbers, hyphens, and underscores", "error")
            return redirect(url_for("index"))
        
        # Check if experiment name already exists
        bucket = storage_client.bucket(BUCKET_NAME)
        prefix = f"{exp_name}/"
        existing_blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
        if existing_blobs:
            flash(f"Experiment name '{exp_name}' already exists. Please choose a different name.", "error")
            return redirect(url_for("index"))
        
        generations = int(generations_str)
        population = int(population_str)
        hidden = int(hidden_str)
        
        if compute_option == "kubernetes":
            success, message = launch_kubernetes_job(exp_name, generations, population, hidden)
            if success:
                flash(
                    f"Experiment '{exp_name}' launched on Kubernetes! Gen: {generations}, Pop: {population}, Hidden: {hidden}",
                    "success"
                )
            else:
                flash(f"Kubernetes launch failed: {message}", "error")
        else:
            # Launch on Cloud Run (existing logic)
            if not CLOUD_FUNCTION_URL:
                flash("Cloud Function URL not configured. Please update CLOUD_FUNCTION_URL in main.py", "error")
                return redirect(url_for("index"))
            
            payload = {
                "exp_name": exp_name,
                "generations": generations,
                "population": population,
                "hidden": hidden,
                "seed": 123,
                "subset_size": 2000
            }
            
            print(f"Launching experiment with payload: {payload}")
            print(f"Calling Cloud Function at: {CLOUD_FUNCTION_URL}")
            
            response = requests.post(
                CLOUD_FUNCTION_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            print(f"Cloud Function response status: {response.status_code}")
            print(f"Cloud Function response: {response.text}")
            
            if response.status_code == 200:
                flash(
                    f"Experiment '{exp_name}' launched on Cloud Run! Gen: {generations}, Pop: {population}, Hidden: {hidden}",
                    "success"
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text
                
                flash(f"Launch failed: {error_msg}", "error")
            
    except requests.exceptions.Timeout:
        print("Request timed out - experiment running in background")
        flash(
            f"Experiment '{exp_name}' started! Gen: {generations}, Pop: {population}, Hidden: {hidden}. Running in background, check back soon.",
            "success"
        )
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
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
    """Delete an experiment from GCS and Kubernetes if applicable"""
    try:
        # Try to delete RayJob if it exists
        delete_rayjob(exp_name)
        
        # Delete from GCS
        num_deleted = delete_experiment_from_gcs(exp_name)
        
        flash(f"Successfully deleted experiment '{exp_name}' ({num_deleted} files)", "success")
    except Exception as e:
        flash(f"Error deleting experiment '{exp_name}': {str(e)}", "error")
    return redirect(url_for("index"))

@app.route("/experiment/<exp_name>")
def experiment(exp_name):
    is_processing = False
    summary = None
    data = None
    first_gen_time = None
    
    try:
        summary_path = f"{exp_name}/final_summary.json"
        summary = load_json_from_gcs(summary_path)
        data = summary["data"]
        
        if data:
            data = {k: v for k, v in data.items() if k not in ['GCS_bucket', 'GCS_prefix']}
            
            if 'total_execution_time' in data:
                data['total_execution_time'] = format_time_duration(data['total_execution_time'])
            
            if 'cloud_run_warmup_time' in data:
                data['cloud_run_warmup_time'] = format_time_duration(data['cloud_run_warmup_time'])
        
        if summary and summary.get("timestamp"):
            summary["timestamp"] = format_timestamp(summary["timestamp"])
            
    except Exception as e:
        is_processing = True
        print(f"No final_summary.json for {exp_name} - experiment is processing")

    df = pd.DataFrame()
    try:
        df = load_history(exp_name)
        if not df.empty and is_processing:
            if "timestamp" in df.columns:
                first_gen_time = df.iloc[0]["timestamp"]
                if isinstance(first_gen_time, str):
                    first_gen_time = first_gen_time.replace("Z", "+00:00")
    except Exception as e:
        print(f"No history yet for {exp_name}: {e}")

    history_plot_html = ""
    time_plot_html = ""

    if not df.empty:
        try:
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