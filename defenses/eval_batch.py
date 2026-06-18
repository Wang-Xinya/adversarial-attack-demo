import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from attacks.fgsm import fgsm_perturb
from attacks.pgd import pgd_perturb
from defenses.random_smooth import random_smooth_predict
from utils.common import save_example_grid, save_robust_curve
from attacks.mi_fgsm import mi_fgsm_perturb

def evaluate_batch(model, loader, device, epsilons,
                   attack_type='pgd', defense_type='random_smooth',
                   defense_params=None, attack_params=None,
                   max_examples=5):
    if defense_params is None:
        defense_params = {'num_samples': 50, 'sigma': 0.1}
    if attack_params is None:
        attack_params = {'pgd_steps': 10, 'alpha_ratio': 0.25, 'random_start': True}

    from utils.evaluate import evaluate_attack_general
    acc_no_defense = []
    examples_no_defense = []
    for eps in epsilons:
        if attack_type == 'fgsm':
            attack_func = fgsm_perturb
            attack_kwargs = {}
        elif attack_type == 'pgd':
            attack_func = pgd_perturb
            alpha = eps * attack_params.get('alpha_ratio', 0.25) if eps > 0 else 0.0
            steps = attack_params.get('pgd_steps', 10) if eps > 0 else 0
            attack_kwargs = {'alpha': alpha, 'steps': steps,
                             'random_start': attack_params.get('random_start', True)}
        elif attack_type == 'mi-fgsm':
            attack_func = mi_fgsm_perturb
            steps = attack_params.get('pgd_steps', 10) if eps > 0 else 0
            alpha = eps / steps if steps > 0 else 0.0
            decay = attack_params.get('decay', 1.0)
            attack_kwargs = {'alpha': alpha, 'steps': steps,
                             'decay': decay,
                             'random_start': attack_params.get('random_start', True)}
        else:
            raise ValueError("Unsupported attack type")
        
        acc, ex = evaluate_attack_general(
            model, loader, device, eps,
            attack_func=attack_func,
            attack_kwargs=attack_kwargs
        )

        acc_no_defense.append(acc)
        examples_no_defense.append(ex)

    acc_with_defense = []
    examples_defense = []

    if defense_type == 'random_smooth':
        from defenses.random_smooth import evaluate_random_smooth
        if attack_type == 'mi-fgsm':
            # 确保 attack_params 包含 decay
            attack_params_with_decay = attack_params.copy()
            attack_params_with_decay['decay'] = attack_params.get('decay', 1.0)
        else:
            attack_params_with_decay = attack_params

        results_defense = evaluate_random_smooth(
            model, loader, device, epsilons,
            num_samples=defense_params['num_samples'],
            sigma=defense_params['sigma'],
            max_examples=max_examples,
            attack_type=attack_type,
            pgd_steps=attack_params['pgd_steps'],
            alpha_ratio=attack_params['alpha_ratio'],
            random_start=attack_params['random_start'],
            decay=attack_params_with_decay.get('decay', 1.0)
        )
        acc_with_defense = [res[1] for res in results_defense]
        examples_defense = [res[2] for res in results_defense]
    elif defense_type == 'None':
        acc_with_defense = acc_no_defense
    else:
        raise ValueError("Wrong defense type")

    fig_examples = _create_example_figure(epsilons, examples_no_defense)

    fig_curve = _create_curve_figure(epsilons, acc_no_defense, acc_with_defense,
                                     attack_type, defense_type)

    fig_diff = _create_diff_figure(epsilons, acc_no_defense, acc_with_defense)

    return fig_examples, fig_curve, fig_diff, acc_no_defense, acc_with_defense

def _create_example_figure(epsilons, examples_per_eps):
    n_eps = len(epsilons)
    n_col = max((len(row) for row in examples_per_eps), default=1)
    fig, axes = plt.subplots(n_eps, n_col, figsize=(2*n_col, 2*n_eps),
                             squeeze=False)
    for i, eps in enumerate(epsilons):
        row = examples_per_eps[i]
        for j in range(len(row)):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"ε={eps}", fontsize=11)
            orig, adv, img = row[j]
            ax.set_title(f"{orig}→{adv}")
            ax.imshow(img, cmap="gray")
        for j in range(len(row), n_col):
            axes[i, j].axis('off')
    plt.tight_layout()
    return fig

def _create_curve_figure(epsilons, acc_no, acc_yes, attack_name, defense_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epsilons, acc_no, 'o-', label=f'No defense ({attack_name})')
    ax.plot(epsilons, acc_yes, 's-', label=f'With {defense_name}')
    ax.set_xlabel('ε (L∞ radius)')
    ax.set_ylabel('Robust accuracy')
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def _create_diff_figure(epsilons, acc_no, acc_yes):
    diffs = [y - x for x, y in zip(acc_no, acc_yes)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(e) for e in epsilons], diffs, color='skyblue')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('ε')
    ax.set_ylabel('Improvement (defense - no defense)')
    ax.set_title('Defense gain')
    fig.tight_layout()
    return fig