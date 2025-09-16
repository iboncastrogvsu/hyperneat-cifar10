import random
import numpy as np
from .cppn import Genome
from .phenotype import Phenotype
from time import time
import ray
import torch

def init_population(pop_size, genome_kwargs=None):
    """
    Initialize a population of random genomes.
    """
    genome_kwargs = genome_kwargs or {}
    return [Genome.random(**genome_kwargs) for _ in range(pop_size)]

def tournament_selection(pop, fitnesses, k=3):
    """
    Select genomes via k-tournament selection.
    """
    selected = []
    n = len(pop)
    for _ in range(n):
        idxs = random.sample(range(n), k)
        best = max(idxs, key=lambda i: fitnesses[i])
        selected.append(pop[best].copy())
    return selected

def reproduce(pop_selected, crossover_rate=0.5, mut_rate=0.2):
    """
    Generate next population by crossover and mutation.
    """
    next_pop = []
    for parent in pop_selected:
        if random.random() < crossover_rate:
            mate = random.choice(pop_selected)
            child = parent.crossover(mate)
        else:
            child = parent.copy()
        child.mutate(rate=mut_rate)
        next_pop.append(child)
    return next_pop

@ray.remote
def evaluate_genome(genome, substrate_cfg, eval_fn, device="cpu"):
    """
    Evaluate a single genome in isolation.
    PyTorch threads limited to 1 to avoid oversubscription.
    """
    torch.set_num_threads(1)
    phen = Phenotype(genome, substrate_cfg)
    return eval_fn(phen, device=device)

def evaluate_population(pop, substrate_cfg, eval_fn, device="cpu", num_cpus=None):
    """
    Evaluate all genomes in parallel using Ray.

    Args:
        pop (list[Genome]): Population to evaluate.
        substrate_cfg (dict): Substrate configuration.
        eval_fn (callable): Fitness evaluation function.
        device (str): Torch device.
        num_cpus (int, optional): Number of CPUs to use in Ray.

    Returns:
        list[float]: Fitness values aligned with population order.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus or None, ignore_reinit_error=True)

    futures = [evaluate_genome.remote(g, substrate_cfg, eval_fn, device=device) for g in pop]
    return ray.get(futures)

def evolve(pop_size,
           substrate_cfg,
           eval_fn,
           generations=10,
           genome_kwargs=None,
           seed=0,
           log_fn=None,
           device="cpu",
           num_cpus=None):
    """
    Run evolutionary loop in parallel using Ray.

    Returns:
        best_genome (Genome): Genome with highest fitness.
        best_fitness (float): Best fitness achieved.
        history (list[dict]): Per-generation statistics.
    """
    random.seed(seed)
    np.random.seed(seed)

    pop = init_population(pop_size, genome_kwargs=genome_kwargs)
    best_genome, best_fitness = None, -1.0
    history = []

    for gen in range(1, generations + 1):
        t0 = time()
        fitnesses = evaluate_population(pop, substrate_cfg, eval_fn, device=device, num_cpus=num_cpus)
        t1 = time()

        avg, mx = float(np.mean(fitnesses)), float(np.max(fitnesses))
        argmx = int(np.argmax(fitnesses))
        history.append({"gen": gen, "avg": avg, "max": mx, "time_s": t1 - t0})
        if log_fn:
            log_fn(history[-1])

        if mx > best_fitness:
            best_fitness, best_genome = mx, pop[argmx].copy()

        selected = tournament_selection(pop, fitnesses, k=3)
        pop = reproduce(selected, crossover_rate=0.6, mut_rate=0.3)

        print(f"Gen {gen:3d} | max {mx:.4f} | avg {avg:.4f} | gen_time {t1-t0:.2f}s")

    if ray.is_initialized():
        ray.shutdown()

    return best_genome, best_fitness, history
