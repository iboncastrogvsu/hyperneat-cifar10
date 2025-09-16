# HyperNEAT on CIFAR-10

This repository implements a **HyperNEAT-style neuroevolution system** applied to the CIFAR-10 image classification dataset.  
It supports both **sequential** (single-process) and **parallel** (Ray-based) evaluation of candidate networks, making it useful for exploring **local vs. distributed/cloud execution trade-offs**.

---

## Repository Structure

hyperneat-cifar10/  
|__ data/ **[CIFAR-10 dataset (downloaded automatically by torchvision)]**  
|__ experiments/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ main.py  **[Entry point, choose backend and run evolution]**  
|__ hyperneat/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ cppn.py  **[Genome definition (CPPN encoding)]**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ evaluate.py  **[Fitness evaluation utilities]**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ evolution_ray.py  **[Evolution loop (Ray parallel backend)]**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ evolution_sequential.py  **[Evolution loop (sequential backend)]**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ phenotype.py  **[Phenotype (PyTorch model) construction from genome]**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ substrate.py  **[Substrate geometry (input, hidden, output positions)]**  
|__ utils/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ dataset.py  **[Dataset loading utilities]**  

---

## Installation and Use

Clone the repository with:

```
git clone https://github.com/iboncastrogvsu/hyperneat-cifar10.git
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### 1. Sequential Evolution  
Runs all genome evaluations on a single process.  

```bash
python -m experiments.main
```  

Set the mode in **experiments/main.py**:

```python
MODE = "sequential"
```  

### 2. Parallel Evolution with Ray  

Evaluates genomes in parallel across available CPU cores.  
Useful for benchmarking local vs. distributed/cloud scaling.  

```bash
python -m experiments.main
``` 

Set the mode in **experiments/main.py**:

```python
MODE = "ray"
```  

You can also control the CPU usage:  

```python
num_cpus = 8
```  

---

## Parameters

You can control the parameters directly via command-line arguments:

| Argument | Default Value | Description
|---|---|---|
| --mode | sequential | Backend mode: sequential (single-process) or ray (parallel evaluation) |
| --cpus | 1 | Number of CPUs to use (if mode=ray). Set higher for parallelism |
| --generations | 15 | Number of evolutionary generations to run |
| --hidden | 16 | Size of hidden substrate (width = height) |
| --population | 15 | Population size (number of genomes per generation) |

--- 
# Author

Ibon Castro Llorente

[Linkedin](https://www.linkedin.com/in/ibon-castro/)