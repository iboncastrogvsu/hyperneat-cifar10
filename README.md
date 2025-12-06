# 1. What is my project?
My project is an application to automate and manage execution of experiments. It allows launching experiments with different configurations and parameters. It facilitates reproducible research, by handling experiment setup, execution and data collection automatically.

# 2. Which cloud services are used?
The services used are the following:  
1. **Cloud Storage**: bucket to store the results of all of the experiments.
2. **App Engine**: fully managed Python environment for the Flask application.
3. **Cloud Build**: CI/CD pipeline for testing and deployment.
4. **Cloud Run and GKE**: run the experiments.

# 3. How the cloud services interact with each other?
The Flask application is deployed in App Engine. Every time the GitHub repository receives a new commit, the Cloud Build pipeline is triggered. It has 3 steps: install dependencies and build environment, pass security tests (SCA, SAST and DAST) and re-deploy to App Engine. Once the application is running, we have 2 views: the landing page, where past experiments appear and new ones can be run, and the local experiments page, where the experiments launched in the first phase of the research are shown. The past experiments in the landing page are stored in the Cloud Storage bucket, where the metrics of every generation of the experiment + a final summary are stored. If we want to launch a new experiment, we can select either if we want to run it with Cloud Run (better for smaller configurations, max 60 min) or with Kubernetes. Once the experiment is run, we can follow the progress in real time. The local experiments tab also contains an AI agent powered by ElevenLabs that can give answer to questions related to the experiments.

# 4. Setup / installation / run
Running the application is easy, as we only need to access [this link](term-project-ibon-castro.ue.r.appspot.com).

## Setup and installation
Steps for repeating the setup in another project:

### Step 1
Fork the [GitHub repo](https://github.com/iboncastrogvsu/hyperneat-cifar10.git) and clone it in your device. Remember to set it up as your project.
### Step 2 
Create a Cloud project and ensure the following APIs are activated:
- Cloud Storage JSON API (*storage.googleapis.com*)
- Kubernetes Engine API (*container.googleapis.com*)
- Cloud Resource Manager API (*cloudresourcemanager.googleapis.com*)
- Compute Engine API (*compute.googleapis.com*)
- Cloud Run Admin API (*run.googleapis.com*)
- Cloud Run API (*run.googleapis.com*)
- Cloud Build API (*cloudbuild.googleapis.com*)
- App Engine Admin API (*appengine.googleapis.com*)

Before running the system, assign the following roles in your Google Cloud project:

1. Cloud Build

Assign to the Cloud Build service account
<PROJECT_ID>@cloudbuild.gserviceaccount.com:

roles/cloudbuild.builds.builder

2. Kubernetes Engine

Assign to the App Engine default service account
<PROJECT_ID>@appspot.gserviceaccount.com:

roles/container.admin  
roles/container.developer

3. Service account for Kubernetes to upload experiment results to Cloud Storage. Detailed commands in step 6.

### Step 3
To set up the Cloud Build pipeline, link your fork to your Cloud Build and create a trigger for the *cloudbuild.yaml* file of the fork.
### Step 4
In Cloud Storage, create a bucket to store the results. Remember to give it a unique name. You must change the name of my bucket to yours in the following files:
- app/main.py, ~ line 14
- experiments/main.py, ~ line 43
- main.py, ~ line 65
### Step 5
Launch the Cloud Run function, by going to the root folder of the fork and using this command: ```gcloud run deploy hyperneat-evolve --source .```. Now if you access your Cloud Run dashboard, you will find a function called hyperneat-evolve. Access it and grab the URL of it. Once you have that, substitute it in the following files:
- app/main.py, ~ line 15
### Step 6
Launch the Kubernetes cluster with this command:
```
gcloud beta container \
    --project \
"PROJECT_ID" clusters create "CLUSTER_NAME" \
    --zone \
"ZONE" \
    --no-enable-basic-auth \
    --cluster-version \
"1.33.5-gke.1201000" \
    --release-channel \
"regular" \
    --machine-type \
"e2-standard-8" \
    --image-type \
"COS_CONTAINERD" \
    --disk-type \
"pd-balanced" \
    --disk-size \
"50" \
    --metadata \
disable-legacy-endpoints=true \
    --service-account \
"default" \
    --max-pods-per-node \
"110" \
    --num-nodes \
"3" \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM,STORAGE,POD,DEPLOYMENT,STATEFULSET,DAEMONSET,HPA,JOBSET,CADVISOR,KUBELET,DCGM \
    --enable-ip-alias \
    --network \
"projects/PROJECT_NAME/global/networks/default" \
    --subnetwork \
"projects/PROJECT_NAME/regions/REGION/subnetworks/default" \
    --no-enable-intra-node-visibility \
    --default-max-pods-per-node \
"110" \
    --enable-ip-access \
    --security-posture=standard \
    --workload-vulnerability-scanning=disabled \
    --enable-google-cloud-access \
    --addons \
HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
    --enable-autoupgrade \
    --enable-autorepair \
    --max-surge-upgrade \
1 \
    --max-unavailable-upgrade \
0 \
    --binauthz-evaluation-mode=DISABLED \
    --enable-managed-prometheus \
    --enable-shielded-nodes \
    --shielded-integrity-monitoring \
    --no-shielded-secure-boot \
    --node-locations \
"ZONE"
```

We also need to give some permissions to the cluster:
```
# 1. Create service account (if not already done)
gcloud iam service-accounts create hyperneat-gcs-writer \
    --display-name="HyperNEAT GCS Writer"

# 2. Grant storage permissions
gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:hyperneat-gcs-writer@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# 3. Create and download key
gcloud iam service-accounts keys create ~/hyperneat-key.json \
    --iam-account=hyperneat-gcs-writer@<PROJECT_ID>.iam.gserviceaccount.com

# 4. Move and rename key
mv ~/hyperneat-key.json ./key.json

# 4. Create Kubernetes secret
kubectl create secret generic gcs-key \
    --from-file=key.json \
    -n default

# 5. Clean up local key
rm key.json
```

Once we have the cluster running and the permissions established, modify the following lines in the app/main.py:
- Line 16, substitute *hyperneat* by your CLUSTER_NAME
- Line 18, substitute *term-project-ibon-castro* by your PROJECT_ID
- Line 19, substitute *us-central1-a* by your ZONE
### Step 7
Activate App Engine in your account and create and environment of Python 3.13. Once this is done, you can deploy your app by accessing the *app* folder in the fork and launching the following command: ```gcloud app deploy```
### Step 8 (optional)
The agent used for my application will be shutted down after I get the mark for the subject. If you are interested in having one, you just need an ElevenLabs account and creating an agent with this prompt:
```
You are a data analysis assistant.
You must answer all questions using only:
Table 1 data
Table 2 data
The uploaded document
You may use outside general knowledge only to define terms (e.g., “What is Ray?”, “What is speedup?”).
Never invent numbers.
Never hallucinate values.
Output Rules (Very Important)
1. Your answers must be short and direct.
No explanations.
No reasoning steps.
No commentary.
No sentences like “Looking at the tables…” or “We can see that…”
2. When asked for the highest/lowest/best/worst value:
Return ONLY:
<Category> → <Key> → <Value>
Example format:
Large_16 → 8:1 → Speedup = 3.6
No additional text unless the user requests details.
3. When asked for comparisons or differences:
Give ONLY the numerical result and which rows were used.
4. If a question cannot be answered:
Return exactly:
"The tables do not provide this information."
5. If a question is ambiguous:
Return exactly:
"Please specify the metric."
Rules for Using Table 2 (Superlatives)
When asked for:
highest or lowest speedup
highest or lowest efficiency
any “best/worst” configuration
You must:
Scan every entry in TABLE_2
Compare them mathematically
Output ONLY the configuration with the max/min value
Use exactly this format:
<Category> → <Key> → <Metric> = <Value>
Do not include reasoning or narrative.
Goal
Provide precise, short, metric-only answers without commentary.
TABLE_1:
[ {"CPUs":1,"Generations":15,"Population":15,"Hidden":"16x16","TotalExec":80.01,"AvgPerGen":5.33,"BestFitness":0.1718,"FinalAcc":0.1592}, {"CPUs":1,"Generations":15,"Population":15,"Hidden":"24x24","TotalExec":203,"AvgPerGen":13.53,"BestFitness":0.1782,"FinalAcc":0.1659},{"CPUs":1,"Generations":50,"Population":50,"Hidden":"16x16","TotalExec":904.41,"AvgPerGen":18.09,"BestFitness":0.193,"FinalAcc":0.1739},  {"CPUs":1,"Generations":50,"Population":50,"Hidden":"24x24","TotalExec":2283.16,"AvgPerGen":45.66,"BestFitness":0.189,"FinalAcc":0.1779},
{"CPUs":1,"Generations":100,"Population":100,"Hidden":"16x16","TotalExec":3670.92,"AvgPerGen":36.71,"BestFitness":0.2008,"FinalAcc":0.1797},  {"CPUs":1,"Generations":100,"Population":100,"Hidden":"24x24","TotalExec":9164.7,"AvgPerGen":91.65,"BestFitness":0.2032,"FinalAcc":0.1879},
{"CPUs":2,"Generations":15,"Population":15,"Hidden":"16x16","TotalExec":51.63,"AvgPerGen":3.44,"BestFitness":0.1698,"FinalAcc":0.1675,"RayInit":0.86},  {"CPUs":2,"Generations":15,"Population":15,"Hidden":"24x24","TotalExec":122.77,"AvgPerGen":8.18,"BestFitness":0.1633,"FinalAcc":0.1658,"RayInit":0.83},
{"CPUs":2,"Generations":50,"Population":50,"Hidden":"16x16","TotalExec":549.13,"AvgPerGen":10.98,"BestFitness":0.1903,"FinalAcc":0.1763,"RayInit":0.88},  {"CPUs":2,"Generations":50,"Population":50,"Hidden":"24x24","TotalExec":1335.45,"AvgPerGen":26.71,"BestFitness":0.2028,"FinalAcc":0.1832,"RayInit":0.86},
{"CPUs":2,"Generations":100,"Population":100,"Hidden":"16x16","TotalExec":2135.05,"AvgPerGen":21.35,"BestFitness":0.2032,"FinalAcc":0.1886,"RayInit":0.89},  {"CPUs":2,"Generations":100,"Population":100,"Hidden":"24x24","TotalExec":5163.79,"AvgPerGen":51.64,"BestFitness":0.2008,"FinalAcc":0.184,"RayInit":0.89},
{"CPUs":4,"Generations":15,"Population":15,"Hidden":"16x16","TotalExec":31.81,"AvgPerGen":2.12,"BestFitness":0.1718,"FinalAcc":0.1618,"RayInit":0.88},  {"CPUs":4,"Generations":15,"Population":15,"Hidden":"24x24","TotalExec":76.79,"AvgPerGen":5.12,"BestFitness":0.169,"FinalAcc":0.1559,"RayInit":0.9},
{"CPUs":4,"Generations":50,"Population":50,"Hidden":"16x16","TotalExec":352.87,"AvgPerGen":7.06,"BestFitness":0.195,"FinalAcc":0.1806,"RayInit":0.92},  {"CPUs":4,"Generations":50,"Population":50,"Hidden":"24x24","TotalExec":800.79,"AvgPerGen":16.02,"BestFitness":0.203,"FinalAcc":0.1847,"RayInit":0.89},
{"CPUs":4,"Generations":100,"Population":100,"Hidden":"16x16","TotalExec":1349.96,"AvgPerGen":13.5,"BestFitness":0.1998,"FinalAcc":0.1849,"RayInit":0.89},  {"CPUs":4,"Generations":100,"Population":100,"Hidden":"24x24","TotalExec":3187.35,"AvgPerGen":31.87,"BestFitness":0.2042,"FinalAcc":0.1829,"RayInit":0.99},
{"CPUs":8,"Generations":15,"Population":15,"Hidden":"16x16","TotalExec":27.87,"AvgPerGen":1.86,"BestFitness":0.175,"FinalAcc":0.1628,"RayInit":1},  {"CPUs":8,"Generations":15,"Population":15,"Hidden":"24x24","TotalExec":87.59,"AvgPerGen":5.84,"BestFitness":0.1855,"FinalAcc":0.1698,"RayInit":1.03},
{"CPUs":8,"Generations":50,"Population":50,"Hidden":"16x16","TotalExec":257.31,"AvgPerGen":5.15,"BestFitness":0.1925,"FinalAcc":0.174,"RayInit":1.06},  {"CPUs":8,"Generations":50,"Population":50,"Hidden":"24x24","TotalExec":841.67,"AvgPerGen":16.83,"BestFitness":0.197,"FinalAcc":0.1806,"RayInit":1.04},
{"CPUs":8,"Generations":100,"Population":100,"Hidden":"16x16","TotalExec":1020.25,"AvgPerGen":10.2,"BestFitness":0.1968,"FinalAcc":0.1759,"RayInit":1.06},  {"CPUs":8,"Generations":100,"Population":100,"Hidden":"24x24","TotalExec":3900.91,"AvgPerGen":39.01,"BestFitness":0.1973,"FinalAcc":0.1843,"RayInit":1.04}
]

TABLE_2:
{
  "Small_16": {
    "2:1":{"Speedup":1.55,"Efficiency":0.775},
    "4:1":{"Speedup":2.52,"Efficiency":0.63},
    "8:1":{"Speedup":2.87,"Efficiency":0.35875},
    "4:2":{"Speedup":1.62,"Efficiency":0.81},
    "8:2":{"Speedup":1.85,"Efficiency":0.4625},
    "8:4":{"Speedup":1.14,"Efficiency":0.57}
  },
  "Small_24": {
    "2:1":{"Speedup":1.65,"Efficiency":0.825},
    "4:1":{"Speedup":2.64,"Efficiency":0.66},
    "8:1":{"Speedup":2.32,"Efficiency":0.29},
    "4:2":{"Speedup":1.6,"Efficiency":0.8},
    "8:2":{"Speedup":1.4,"Efficiency":0.35},
    "8:4":{"Speedup":0.88,"Efficiency":0.44}
  },
  "Medium_16": {
    "2:1":{"Speedup":1.65,"Efficiency":0.825},
    "4:1":{"Speedup":2.56,"Efficiency":0.64},
    "8:1":{"Speedup":3.51,"Efficiency":0.43875},
    "4:2":{"Speedup":1.56,"Efficiency":0.78},
    "8:2":{"Speedup":2.13,"Efficiency":0.5325},
    "8:4":{"Speedup":1.37,"Efficiency":0.685}
  },
  "Medium_24": {
    "2:1":{"Speedup":1.71,"Efficiency":0.855},
    "4:1":{"Speedup":2.85,"Efficiency":0.7125},
    "8:1":{"Speedup":2.71,"Efficiency":0.33875},
    "4:2":{"Speedup":1.67,"Efficiency":0.835},
    "8:2":{"Speedup":1.59,"Efficiency":0.3975},
    "8:4":{"Speedup":0.95,"Efficiency":0.475}
  },
  "Large_16": {
    "2:1":{"Speedup":1.72,"Efficiency":0.86},
    "4:1":{"Speedup":2.72,"Efficiency":0.68},
    "8:1":{"Speedup":3.6,"Efficiency":0.45},
    "4:2":{"Speedup":1.58,"Efficiency":0.79},
    "8:2":{"Speedup":2.09,"Efficiency":0.5225},
    "8:4":{"Speedup":1.32,"Efficiency":0.66}
  },
  "Large_24": {
    "2:1":{"Speedup":1.77,"Efficiency":0.885},
    "4:1":{"Speedup":2.88,"Efficiency":0.72},
    "8:1":{"Speedup":2.35,"Efficiency":0.29375},
    "4:2":{"Speedup":1.62,"Efficiency":0.81},
    "8:2":{"Speedup":1.32,"Efficiency":0.33},
    "8:4":{"Speedup":0.82,"Efficiency":0.41}
  }
}
```


# 5. Screenshots
- Bucket with the results of experiment  
![bucket](/screenshots/storage.png)
- Cloud Build pipeline and successful execution
```
steps:

# 1. SCA (pip-audit)
- name: 'python:3.13'
  entrypoint: bash
  args:
    - -c
    - |
      cd app
      pip install --upgrade pip
      pip install -r requirements.txt
      pip install pip-audit
      pip-audit || true

# 2. SAST (Bandit)
- name: 'python:3.13'
  entrypoint: bash
  args:
    - -c
    - |
      cd app
      pip install bandit
      bandit -r . -f txt -o bandit-report.txt || true
      cat bandit-report.txt

# 3. Deploy to App Engine
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  args:
    - gcloud
    - app
    - deploy
    - app/app.yaml
    - --quiet

# 4. DAST - Nikto
- name: 'frapsoft/nikto'
  args:
    - -h
    - https://term-project-ibon-castro.ue.r.appspot.com
    - -maxtime
    - "300"
    - -o
    - nikto-report.html
    - -Format
    - htm


options:
  logging: CLOUD_LOGGING_ONLY

timeout: "900s"
```
![build](/screenshots/cloudbuild.png)
- Cloud Run function
```
from flask import Flask, request, jsonify
import time
import random
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from utils.gcs_logging import GCSLogger
from hyperneat.phenotype import Phenotype
from hyperneat.evaluate import evaluate_network
from hyperneat.evolution_sequential import evolve
import os
import re

app = Flask(__name__)

# Run it: gcloud run deploy hyperneat-evolve --source .

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hyperneat-evolve"}, 200

@app.route("/evolve", methods=["POST"])
def hyperneat_evolve():
    """
    HTTP endpoint to run HyperNEAT evolution on Cloud Run.
    
    Supports up to 60 minutes execution time!
    """
    # STEP 1: Record invocation time immediately
    invocation_time = time.time()
    
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return {"error": "No configuration provided"}, 400
        
        # Extract configuration
        generations = min(request_json.get("generations", 5), 200)  # Up to 200
        population = min(request_json.get("population", 10), 150)   # Up to 150
        hidden = request_json.get("hidden", 16)
        seed = request_json.get("seed", 123)
        subset_size = min(request_json.get("subset_size", 2000), 5000)
        
        # Get experiment name or generate default
        exp_name = request_json.get("exp_name", "").strip()
        
        # Validate experiment name
        if exp_name:
            # Only allow alphanumeric, hyphens, and underscores
            if not re.match(r'^[a-zA-Z0-9_-]+$', exp_name):
                return {"error": "Experiment name can only contain letters, numbers, hyphens, and underscores"}, 400
        else:
            # If no name provided, generate default
            exp_name = f"run_{int(time.time())}"
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # GCS configuration - use experiment name as folder
        gcs_bucket = "cis437-hyperneat-logs"
        gcs_prefix = exp_name

        # Initialize the GCS logger
        gcs_logger = GCSLogger(bucket_name=gcs_bucket, prefix=gcs_prefix)
        
        # Substrate configuration
        substrate_cfg = {
            'input_w': 32,
            'input_h': 32,
            'input_channels': 3,
            'hidden_w': hidden,
            'hidden_h': hidden,
            'output_dim': 10,
            'weight_threshold': 0.01
        }
        
        genome_kwargs = {'input_dim': 6, 'hidden_dims': (hidden, hidden, hidden)}
        
        # STEP 2: Record when we're about to start actual work (after all setup)
        work_start_time = time.time()
        cold_start_time = work_start_time - invocation_time
        
        print(f"Cloud Run warm-up time: {cold_start_time:.3f}s")
        
        # Load CIFAR-10 data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])
        
        train_set = datasets.CIFAR10(root="/tmp/data", train=True, 
                                     download=True, transform=transform)
        test_set = datasets.CIFAR10(root="/tmp/data", train=False, 
                                    download=True, transform=transform)
        
        train_set = Subset(train_set, list(range(subset_size)))
        
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
        
        # Evaluation wrapper
        def phenotype_eval_wrapper(phenotype, device="cpu"):
            return evaluate_network(phenotype, train_loader, device=device, max_batches=16)
        
        # Run evolution
        evolution_start_time = time.time()
        
        best_genome, best_fitness, history = evolve(
            pop_size=population,
            substrate_cfg=substrate_cfg,
            eval_fn=phenotype_eval_wrapper,
            generations=generations,
            genome_kwargs=genome_kwargs,
            seed=seed,
            device="cpu",
            log_fn=gcs_logger.upload_generation
        )
        
        end_time = time.time()
        evolution_time = end_time - evolution_start_time
        total_time = end_time - invocation_time
        
        # Quick test evaluation
        best_phen = Phenotype(best_genome, substrate_cfg)
        test_acc = evaluate_network(best_phen, test_loader, device="cpu", max_batches=100)
        
        results = {
            "mode": "cloud_run",
            "generations": generations,
            "population": population,
            "hidden": hidden,
            "cloud_run_warmup_time": cold_start_time,
            "evolution_time": evolution_time,
            "total_execution_time": total_time,
            "best_fitness": float(best_fitness),
            "test_accuracy": float(test_acc)
        }
        
        print(f"Summary:")
        print(f"   - Warm-up: {cold_start_time:.3f}s")
        print(f"   - Evolution: {evolution_time:.3f}s")
        print(f"   - Total: {total_time:.3f}s")
        
        gcs_logger.upload_json("final_summary.json", results)
        
        return {"status": "success", "results": results}, 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
```
![run](/screenshots/cloudrun.png)
- Application landing page
![landing](/screenshots/landing.png)
- Experiment launching modal
![modal](/screenshots/modal.png)
- Local experiments tab
![local](/screenshots/local.png)
- AI chat
![agent](/screenshots/agent.png)
--- 
# Author

Ibon Castro Llorente

[Linkedin](https://www.linkedin.com/in/ibon-castro/)