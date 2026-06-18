import torch
import torch.nn.functional as F

def fgsm_perturb(model, x, y, epsilon):
    if epsilon <= 0:
        return x.detach().clamp(0, 1)
    x = x.detach().clone().requires_grad_(True)
    out = model(x)
    loss = F.nll_loss(out, y)
    (grad,) = torch.autograd.grad(loss, x)
    x_adv = x.detach() + epsilon * grad.sign()
    return x_adv.clamp(0, 1)