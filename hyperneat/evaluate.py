import torch
import torch.nn.functional as F

def evaluate_network(model, dataloader, device="cpu", max_batches=32):
    """
    Evaluate a phenotype (PyTorch model) on a subset of the dataset.

    Args:
        model (torch.nn.Module): Neural network phenotype to evaluate.
        dataloader (DataLoader): PyTorch dataloader providing batches of (x, y).
        device (str): Device for evaluation ("cpu" or "cuda").
        max_batches (int): Number of batches to evaluate for efficiency.

    Returns:
        float: Accuracy of the model over the evaluated subset.
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if i + 1 >= max_batches:  # Early stop for speed
                break
    return correct / total if total > 0 else 0.0
