import numpy as np
import torch

from .metrics import compute_metrics


@torch.no_grad()
def predict(net, loader, device):
    """net: the CLIPModel (maps [N,3,H,W] -> [N,1]). Averages predictions over the crops of each image."""
    net.eval()
    preds, targets = [], []
    for x, y in loader:
        b, n = x.shape[:2]
        out = net(x.flatten(0, 1).to(device, non_blocking=True))   # [b*n, 1]
        preds.append(out.view(b, n).float().mean(1).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate(net, loaders: dict, device):
    """{name: metrics} for every loader."""
    results = {}
    for name, loader in loaders.items():
        p, t = predict(net, loader, device)
        results[name] = compute_metrics(p, t)
    return results


def mean_of(results: dict, key: str) -> float:
    return float(np.mean([m[key] for m in results.values()]))
