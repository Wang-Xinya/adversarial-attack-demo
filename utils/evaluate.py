import torch
from typing import Callable, List, Tuple, Any

def evaluate_attack_general(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epsilon: float,
    attack_func: Callable,
    attack_kwargs: dict = None,
    max_examples: int = 5,
) -> Tuple[float, List[Tuple[int, int, Any]]]:
    model.eval()
    robust_correct = 0
    eligible = 0
    examples = []
    attack_kwargs = attack_kwargs or {}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.inference_mode():
            clean_pred = model(x).argmax(dim=1)
        mask = clean_pred == y
        if not mask.any():
            continue
        x_sub = x[mask].detach()
        y_sub = y[mask]

        x_adv = attack_func(model, x_sub, y_sub, epsilon, **attack_kwargs)

        with torch.inference_mode():
            adv_pred = model(x_adv).argmax(dim=1)

        robust_correct += (adv_pred == y_sub).sum().item()
        eligible += y_sub.size(0)

        if len(examples) < max_examples:
            clean_pred_sub = clean_pred[mask].cpu()
            for k in range(y_sub.size(0)):
                if len(examples) >= max_examples:
                    break
                still_ok = adv_pred[k].item() == y_sub[k].item()
                if epsilon > 0 and still_ok:
                    continue
                examples.append((
                    int(clean_pred_sub[k]),
                    int(adv_pred[k].item()),
                    x_adv[k, 0].cpu().numpy()
                ))
    acc = robust_correct / eligible if eligible else 0.0
    return acc, examples