import torch
import torch.nn.functional as F

def project_linf(x_adv, x0, epsilon):
    delta = (x_adv - x0).clamp(min=-epsilon, max=epsilon)
    return (x0 + delta).clamp(0, 1)

def pgd_perturb(model, x0, y, epsilon, alpha, steps, random_start=True):
    if epsilon <= 0 or steps <= 0:
        return x0.detach().clamp(0, 1)
    x0 = x0.detach()
    if random_start:
        noise = (torch.rand_like(x0) * 2 - 1) * epsilon
        x = (x0 + noise).clamp(0, 1)
    else:
        x = x0.clone()
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        out = model(x)
        loss = F.nll_loss(out, y)
        (grad,) = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * grad.sign()
        x = project_linf(x, x0, epsilon)
    return x.detach()