import torch
import torch.nn.functional as F

def mi_fgsm_perturb(model, x, y, epsilon, alpha=None, steps=10, decay=1.0, random_start=True):

    if epsilon <= 0 or steps <= 0:
        return x.detach().clamp(0, 1)

    x0 = x.detach()
    if alpha is None:
        alpha = epsilon / steps

    if random_start:
        noise = (torch.rand_like(x0) * 2 - 1) * epsilon
        x_adv = (x0 + noise).clamp(0, 1)
    else:
        x_adv = x0.clone()

    momentum = torch.zeros_like(x0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        out = model(x_adv)
        loss = F.nll_loss(out, y)
        (grad,) = torch.autograd.grad(loss, x_adv)

        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=1, dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1) + 1e-10
        grad = grad / grad_norm

        momentum = decay * momentum + grad

        x_adv = x_adv.detach() + alpha * momentum.sign()

        delta = (x_adv - x0).clamp(min=-epsilon, max=epsilon)
        x_adv = (x0 + delta).clamp(0, 1)

    return x_adv.detach()