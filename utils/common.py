import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)

def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_robust_curve(epsilons, accuracies, title, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.05, step=0.1))
    plt.xlabel("ε (L∞ radius)")
    plt.ylabel("Robust accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_example_grid(epsilons, examples_per_eps, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_eps = len(epsilons)
    n_col = max((len(row) for row in examples_per_eps), default=1)
    plt.figure(figsize=(2 * n_col, 2 * n_eps))
    cnt = 0
    for i, eps in enumerate(epsilons):
        row = examples_per_eps[i]
        for j in range(len(row)):
            cnt += 1
            ax = plt.subplot(n_eps, n_col, cnt)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"ε={eps}", fontsize=11)
            orig, adv, img = row[j]
            ax.set_title(f"{orig} → {adv}")
            ax.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def get_single_image(loader, index):
    for i, (x, y) in enumerate(loader):
        if i * loader.batch_size <= index < (i+1) * loader.batch_size:
            pos = index - i * loader.batch_size
            return x[pos].unsqueeze(0), y[pos].item()
    raise IndexError("index out of range")

def get_prediction(model, x, device):
    model.eval()
    with torch.inference_mode():
        x = x.to(device)
        logits = model(x)
        probs = torch.exp(logits)  # log_softmax -> softmax
        conf, pred = probs.max(dim=1)
    return pred.item(), conf.item()