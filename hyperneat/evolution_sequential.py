import random
import numpy as np
from .cppn import Genome
from .phenotype import Phenotype
from time import time

def init_population(pop_size, genome_kwargs=None):
    """
    Initialize a population of random genomes.
    """
    genome_kwargs = genome_kwargs or {}
    return [Genome.random(**genome_kwargs) for _ in range(pop_size)]

def evaluate_population(pop, substrate_cfg, eval_fn, device="cpu"):
    """
    Evaluate the fitness of each genome sequentially.

    Args:
        pop (list[Genome]): Current population.
        substrate_cfg (dict): Substrate configuration.
        eval_fn (callable): Function that accepts a Phenotype and returns fitness.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        list[float]: Fitness values aligned with population order.
    """
    fitnesses = []
    for g in pop:
        phen = Phenotype(g, substrate_cfg)
        f = eval_fn(phen, device=device)
        fitnesses.append(f)
    return fitnesses

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

def evolve(pop_size,
           substrate_cfg,
           eval_fn,
           generations=10,
           genome_kwargs=None,
           seed=0,
           log_fn=None,
           device="cpu",
           **kwargs):
    """
    Run evolutionary loop sequentially.

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
        fitnesses = evaluate_population(pop, substrate_cfg, eval_fn, device=device)
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

    return best_genome, best_fitness, history
