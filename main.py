import functions_framework
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

@functions_framework.http
def hyperneat_evolve(request):
    """
    HTTP Cloud Function to run HyperNEAT evolution.
    
    WARNING: Limited to 60 minutes maximum execution time!
    """
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return {"error": "No configuration provided"}, 400
        
        # Extract configuration with strict limits for Cloud Functions
        generations = min(request_json.get("generations", 5), 10)
        population = min(request_json.get("population", 10), 20)  # Max 20 population
        hidden = request_json.get("hidden", 16)
        seed = request_json.get("seed", 123)
        subset_size = min(request_json.get("subset_size", 2000), 2000)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # GCS configuration
        gcs_bucket = "cis437-hyperneat-logs"
        gcs_prefix = f"cf_run_{int(time.time())}"

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
        
        # Run evolution (sequential only - Ray not supported in Cloud Functions)
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
            "mode": "cloud_function",
            "generations": generations,
            "population": population,
            "hidden": hidden,
            "total_execution_time": total_time,
            "best_fitness": float(best_fitness),
            "test_accuracy": float(test_acc),
            "gcs_bucket": gcs_bucket,
            "gcs_prefix": gcs_prefix
        }
        
        gcs_logger.upload_json("final_summary.json", results)
        
        return {"status": "success", "results": results}, 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}, 500