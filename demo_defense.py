#!/usr/bin/env python3
"""
演示随机平滑防御在 MNIST + LeNet 上的效果。
使用 PGD 攻击（默认）生成对抗样本，然后采用随机平滑进行预测，
统计鲁棒准确率，并绘制曲线和示例图。
"""

from pathlib import Path
import torch
from models.lenet import MNISTLeNet
from data.mnist import get_mnist_loader
from utils.common import set_seed, resolve_device, save_robust_curve, save_example_grid
from utils.train import prepare_checkpoint, load_lenet_state_dict
from defenses.random_smooth import evaluate_random_smooth


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

# ---------- 随机平滑参数 ----------
NUM_SAMPLES = 100            
SIGMA = 0.25                 
MAX_EXAMPLES = 5          

# ---------- 攻击参数 ----------
PGD_STEPS = 10
ALPHA_RATIO = 0.25         
RANDOM_START = True         

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

    print("\n=== 随机平滑防御评估（使用 PGD 攻击）===")
    print(f"参数：采样次数={NUM_SAMPLES}, 噪声标准差={SIGMA}")
    print(f"PGD 步数={PGD_STEPS}, 步长比例={ALPHA_RATIO}, 随机起点={RANDOM_START}")

    results = evaluate_random_smooth(
        model=model,
        loader=loader,
        device=device,
        epsilons=EPSILONS,
        num_samples=NUM_SAMPLES,
        sigma=SIGMA,
        max_examples=MAX_EXAMPLES,
        attack_type='fgsm',  # or 'pgd'
        pgd_steps=PGD_STEPS,
        alpha_ratio=ALPHA_RATIO,
        random_start=RANDOM_START
    )

    accuracies = []
    all_examples = []
    for eps, acc, ex in results:
        accuracies.append(acc)
        all_examples.append(ex)
        print(f"ε={eps:.2f} -> 鲁棒准确率 = {acc:.4f}")

    save_robust_curve(
        EPSILONS,
        accuracies,
        f"Random Smooth Defense (samples={NUM_SAMPLES}, σ={SIGMA})",
        OUTPUT_DIR / "defense_random_smooth_effect.png"
    )

    save_example_grid(
        EPSILONS,
        all_examples,
        OUTPUT_DIR / "defense_random_smooth_examples.png"
    )

    print(f"\n完成！图像保存在：")
    print(f"  - {OUTPUT_DIR / 'defense_random_smooth_effect.png'}")
    print(f"  - {OUTPUT_DIR / 'defense_random_smooth_examples.png'}")

if __name__ == "__main__":
    main()