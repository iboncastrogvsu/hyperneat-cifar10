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
        start_time = time.time()
        
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
        total_time = end_time - start_time
        
        # Quick test evaluation
        best_phen = Phenotype(best_genome, substrate_cfg)
        test_acc = evaluate_network(best_phen, test_loader, device="cpu", max_batches=100)
        
        results = {
            "mode": "cloud_run",
            "generations": generations,
            "population": population,
            "hidden": hidden,
            "total_execution_time": total_time,
            "best_fitness": float(best_fitness),
            "test_accuracy": float(test_acc)
        }
        
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