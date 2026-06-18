import torch
from attacks.fgsm import fgsm_perturb
from attacks.pgd import pgd_perturb
from utils.common import get_prediction
from attacks.mi_fgsm import mi_fgsm_perturb

def single_attack(model, x, y, device, attack_type='fgsm', epsilon=0.1, **kwargs):
    x = x.to(device)
    y = torch.tensor([y], device=device)

    pred_orig, conf_orig = get_prediction(model, x, device)

    if attack_type == 'fgsm':
        x_adv = fgsm_perturb(model, x, y, epsilon)
    elif attack_type == 'pgd':
        alpha = kwargs.get('alpha', epsilon * 0.25)
        steps = kwargs.get('steps', 10)
        random_start = kwargs.get('random_start', True)
        x_adv = pgd_perturb(model, x, y, epsilon, alpha, steps, random_start)
    elif attack_type == 'mi-fgsm':
        steps = kwargs.get('steps', 10)
        alpha = kwargs.get('alpha', None)
        decay = kwargs.get('decay', 1.0)
        random_start = kwargs.get('random_start', True)
        x_adv = mi_fgsm_perturb(model, x, y, epsilon, alpha, steps, decay, random_start)
    else:
        raise ValueError("attack_type must be 'fgsm' or 'pgd' or 'mi-fgsm'")

    pred_adv, conf_adv = get_prediction(model, x_adv, device)

    original = x.cpu()
    adversarial = x_adv.cpu()
    perturbation = adversarial - original

    return {
        'original': original,
        'adversarial': adversarial,
        'perturbation': perturbation,
        'orig_label': y.item(),
        'orig_pred': pred_orig,
        'orig_conf': conf_orig,
        'adv_pred': pred_adv,
        'adv_conf': conf_adv
    }