#!/usr/bin/env python3
from pathlib import Path
import torch
from models.lenet import MNISTLeNet
from data.mnist import get_mnist_loader
from utils.common import set_seed, resolve_device, save_robust_curve, save_example_grid
from utils.train import prepare_checkpoint, load_lenet_state_dict
from utils.evaluate import evaluate_attack_general
from attacks.fgsm import fgsm_perturb
from attacks.pgd import pgd_perturb

DATA_DIR = Path("./data")
CHECKPOINT = Path("./lenet_mnist_model.pth")
BATCH_SIZE = 256
DEVICE = "auto"
SEED = 0
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
OUTPUT_DIR = Path("./results")
NUM_WORKERS = 0
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
FORCE_TRAIN = False

# PGD
PGD_STEPS = 10
ALPHA_RATIO = 0.25

def main():
    set_seed(SEED)
    device = resolve_device(DEVICE)

    prepare_checkpoint(
        CHECKPOINT, DATA_DIR, device,
        train_epochs=TRAIN_EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        force_train=FORCE_TRAIN
    )
    model = MNISTLeNet().to(device)
    load_lenet_state_dict(model, CHECKPOINT, device)
    model.eval()

    loader = get_mnist_loader(DATA_DIR, BATCH_SIZE, train=False, num_workers=NUM_WORKERS)

    # ====== FGSM ======
    print("\n=== FGSM Attack ===")
    fgsm_accuracies = []
    fgsm_examples = []
    for eps in EPSILONS:
        acc, ex = evaluate_attack_general(
            model, loader, device, eps,
            attack_func=fgsm_perturb,
            attack_kwargs={}
        )
        fgsm_accuracies.append(acc)
        fgsm_examples.append(ex)
        print(f"ε={eps:.2f} -> robust acc = {acc:.4f}")

    save_robust_curve(EPSILONS, fgsm_accuracies, "FGSM Robust Accuracy", OUTPUT_DIR / "FGSM_attack_effect.png")
    save_example_grid(EPSILONS, fgsm_examples, OUTPUT_DIR / "FGSM_attack_examples.png")

    # ====== PGD ======
    print("\n=== PGD Attack ===")
    pgd_accuracies = []
    pgd_examples = []
    for eps in EPSILONS:
        alpha = eps * ALPHA_RATIO if eps > 0 else 0.0
        steps = PGD_STEPS if eps > 0 else 0
        acc, ex = evaluate_attack_general(
            model, loader, device, eps,
            attack_func=pgd_perturb,
            attack_kwargs={'alpha': alpha, 'steps': steps, 'random_start': True}
        )
        pgd_accuracies.append(acc)
        pgd_examples.append(ex)
        print(f"ε={eps:.2f} -> robust acc = {acc:.4f}")

    save_robust_curve(EPSILONS, pgd_accuracies, "PGD Robust Accuracy", OUTPUT_DIR / "PGD_attack_effect.png")
    save_example_grid(EPSILONS, pgd_examples, OUTPUT_DIR / "PGD_attack_examples.png")

    print(f"\nDone. Images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()