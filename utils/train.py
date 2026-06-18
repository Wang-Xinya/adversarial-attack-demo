import torch
import torch.nn.functional as F
from pathlib import Path
from models.lenet import MNISTLeNet
from data.mnist import get_mnist_loader

def train_lenet_and_save(checkpoint: Path, data_dir: Path, device: torch.device,
                         epochs: int, batch_size: int, num_workers: int):
    train_loader = get_mnist_loader(data_dir, batch_size, train=True, num_workers=num_workers)
    model = MNISTLeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print(f"[train] training for {epochs} epochs → {checkpoint.resolve()} ...")
    for epoch in range(epochs):
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.nll_loss(logits, y)
            loss.backward()
            opt.step()
            with torch.inference_mode():
                pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"  epoch {epoch+1}/{epochs}  train acc = {correct/total:.4f}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"[train] saved to {checkpoint.resolve()}")

def prepare_checkpoint(checkpoint: Path, data_dir: Path, device: torch.device,
                       *, train_epochs: int, train_batch_size: int,
                       num_workers: int, force_train: bool):
    from data.mnist import ensure_mnist_downloaded
    ensure_mnist_downloaded(data_dir)
    if checkpoint.is_file() and not force_train:
        print(f"[model] using existing {checkpoint.resolve()}")
        return
    if force_train and checkpoint.is_file():
        print("[model] FORCE_TRAIN=True, retraining")
    train_lenet_and_save(checkpoint, data_dir, device, train_epochs,
                         train_batch_size, num_workers)

def load_lenet_state_dict(model: torch.nn.Module, path: Path, map_location: torch.device):
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)