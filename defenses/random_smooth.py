import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Any

def random_smooth_predict(model, x, num_samples=50, sigma=0.1, device=None):
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    model.eval()

    if x.dim() == 4 and x.size(0) == 1:
        x = x.squeeze(0)
    elif x.dim() != 3:
        raise ValueError("Input must be a single sample (batch size 1)")

    with torch.inference_mode():
        x_expanded = x.unsqueeze(0).expand(num_samples, *x.shape)
        noise = torch.randn_like(x_expanded) * sigma
        noisy_inputs = (x_expanded + noise).clamp(0, 1)
        batch_size = 64
        preds = []
        for i in range(0, num_samples, batch_size):
            batch = noisy_inputs[i:i+batch_size]
            logits = model(batch)
            pred = logits.argmax(dim=1)
            preds.append(pred)
        preds = torch.cat(preds, dim=0)
        counts = torch.bincount(preds)
        voted_label = counts.argmax().item()
        confidence = counts[voted_label].item() / num_samples
    return voted_label, confidence

def evaluate_random_smooth(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epsilons: List[float],
    num_samples: int = 50,
    sigma: float = 0.1,
    max_examples: int = 5,
    attack_type: str = 'pgd',      # 'fgsm' 或 'pgd'
    pgd_steps: int = 10,
    alpha_ratio: float = 0.25,
    random_start: bool = True,
    decay: float = 1.0
) -> List[Tuple[float, float, List[Tuple[int, int, Any]]]]:
    if attack_type == 'fgsm':
        from attacks.fgsm import fgsm_perturb
        def attack_wrapper(model, x, y, eps, **kwargs):
            return fgsm_perturb(model, x, y, eps)
    elif attack_type == 'pgd':
        from attacks.pgd import pgd_perturb
        def attack_wrapper(model, x, y, eps, **kwargs):
            alpha = eps * alpha_ratio if eps > 0 else 0.0
            steps = pgd_steps if eps > 0 else 0
            return pgd_perturb(model, x, y, eps, alpha, steps, random_start)
    elif attack_type == 'mi-fgsm':
        from attacks.mi_fgsm import mi_fgsm_perturb
        def attack_wrapper(model, x, y, eps, **kwargs):
            steps = pgd_steps if eps > 0 else 0 
            alpha = eps / steps if steps > 0 else 0.0
            return mi_fgsm_perturb(model, x, y, eps, alpha, steps, decay, random_start)
    else:
        raise ValueError("attack_type must be 'fgsm' or 'pgd' or 'mi-fgsm'")

    model.eval()
    results = []

    for eps in epsilons:
        correct = 0
        total = 0
        examples = []
        print(f"  评估 ε={eps:.2f} ...")
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.inference_mode():
                clean_pred = model(x).argmax(dim=1)
            mask = clean_pred == y
            if not mask.any():
                continue
            x_sub = x[mask]
            y_sub = y[mask]

            if eps > 0:
                x_adv = attack_wrapper(model, x_sub, y_sub, eps)
            else:
                x_adv = x_sub.clone()

            for k in range(x_adv.size(0)):
                pred_label, _ = random_smooth_predict(
                    model, x_adv[k:k+1], num_samples, sigma, device
                )
                if pred_label == y_sub[k].item():
                    correct += 1
                total += 1

                if len(examples) < max_examples and eps > 0 and pred_label != y_sub[k].item():
                    examples.append((
                        int(y_sub[k].item()),
                        pred_label,
                        x_adv[k, 0].cpu().numpy()
                    ))

        acc = correct / total if total > 0 else 0.0
        results.append((eps, acc, examples))
        print(f"    ε={eps:.2f} 鲁棒准确率 = {acc:.4f} (正确 {correct}/{total})")

    return results