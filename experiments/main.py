import random
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import argparse

from hyperneat.phenotype import Phenotype
from hyperneat.evaluate import evaluate_network

# ---------------------------
# CLI arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Run HyperNEAT Evolutionary Experiment")
parser.add_argument("--mode", type=str, default="sequential", help="Sequential or ray (parallel)")
parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs (1 = sequential, >1 = Ray)")
parser.add_argument("--generations", type=int, default=15, help="Number of generations")
parser.add_argument("--hidden", type=int, default=16, help="Hidden substrate size (width=height)")
parser.add_argument("--population", type=int, default=15, help="Population size")
args = parser.parse_args()

# ---------------------------
# Choose evolution backend
# ---------------------------
if args.mode == "sequential":
    from hyperneat.evolution_sequential import evolve
    MODE = "sequential"
else:
    from hyperneat.evolution_ray import evolve
    MODE = "ray"

# ---------------------------
# HyperNEAT configuration
# ---------------------------
substrate_cfg = {
    'input_w': 32,
    'input_h': 32,
    'input_channels': 3,
    'hidden_w': args.hidden,
    'hidden_h': args.hidden,
    'output_dim': 10,
    'weight_threshold': 0.01
}

# Adjust CPPN hidden_dims based on substrate size
if args.hidden <= 16:
    genome_kwargs = {'input_dim': 6, 'hidden_dims': (24, 24, 12)}
else:
    genome_kwargs = {'input_dim': 6, 'hidden_dims': (32, 32, 16)}

# ---------------------------
# CIFAR-10 data
# ---------------------------
def get_cifar10_loaders(batch_size=128, subset_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    if subset_size is not None:
        train_set = Subset(train_set, list(range(min(subset_size, len(train_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

# ---------------------------
# Evaluation wrapper
# ---------------------------
train_loader_for_eval = None
def phenotype_eval_wrapper(phenotype, device="cpu"):
    global train_loader_for_eval
    return evaluate_network(phenotype, train_loader_for_eval, device=device, max_batches=32)

# ---------------------------
# Run evolution
# ---------------------------
if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128, subset_size=4000)
    train_loader_for_eval = train_loader

    def log_fn(entry):
        print("LOG:", entry)

    print(f"Starting evolution (mode={MODE}, pop={args.population}, gens={args.generations}, hidden={args.hidden}, cpus={args.cpus})...")
    best_genome, best_fitness, history = evolve(
        pop_size=args.population,
        substrate_cfg=substrate_cfg,
        eval_fn=phenotype_eval_wrapper,
        generations=args.generations,
        genome_kwargs=genome_kwargs,
        seed=seed,
        log_fn=log_fn,
        device="cpu",
        num_cpus=args.cpus if MODE == "ray" else None
    )

    print(f"Finished. Best fitness (train subset): {best_fitness:.4f}")

    best_phen = Phenotype(best_genome, substrate_cfg)
    test_acc = evaluate_network(best_phen, test_loader, device="cpu", max_batches=400)
    print(f"Test accuracy (estimated on test set): {test_acc:.4f}")
