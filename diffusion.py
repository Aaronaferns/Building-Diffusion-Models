import torch
import tqdm

NUM_TIMESTEPS = 1000

BETA__T = torch.linspace(1e-4, 0.02, NUM_TIMESTEPS)
ALPHA__T = 1.0 - BETA__T
ALPHA_BAR__T = torch.cumprod(ALPHA__T, dim=0)

def forwardDiffusion(X0__BCHW):
    device = X0__BCHW.device
    beta_T = BETA__T.to(device)
    alpha_T = ALPHA__T.to(device)
    alpha_bar_T = ALPHA_BAR__T.to(device)

    B = X0__BCHW.shape[0]
    eps__BCHW = torch.randn_like(X0__BCHW)
    t__B = torch.randint(low=1, high=NUM_TIMESTEPS+1, size=(B,), device=device, dtype=torch.long) # t in [1,1000]

    ab = alpha_bar_T[t__B - 1][:, None, None, None] #because of 0 indicing
    Xt_BCHW = torch.sqrt(ab) * X0__BCHW + torch.sqrt(1.0 - ab) * eps__BCHW
    return eps__BCHW, t__B, Xt_BCHW

def diffusionTrainStep(model,ema, optimizer, loss_fn, X0, device):
    X0__BCHW = X0.to(device)
    eps__BCHW, t__B, Xt_BCHW = forwardDiffusion(X0__BCHW)
    eps_pred__BCHW = model(Xt_BCHW, t__B)
    loss = loss_fn(eps_pred__BCHW, eps__BCHW)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema.update(model)
    return loss.item()

@torch.no_grad()
def sample(model, B, C, H, W, device=None):
    model.eval()
    device = next(model.parameters()).device if device is None else device

    beta_T = BETA__T.to(device)
    alpha_T = ALPHA__T.to(device)
    alpha_bar_T = ALPHA_BAR__T.to(device)

    x = torch.randn((B, C, H, W), device=device)

    for t in range(NUM_TIMESTEPS, 0, -1):
        t__B = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t__B)

        alpha_t = alpha_T[t__B - 1][:, None, None, None]
        beta_t  = beta_T[t__B - 1][:, None, None, None]
        ab_t    = alpha_bar_T[t__B - 1][:, None, None, None]

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_pred)

        if t > 1:
            ab_prev = alpha_bar_T[t__B - 2][:, None, None, None]
            beta_tilde = ((1.0 - ab_prev) / (1.0 - ab_t)) * beta_t
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_tilde) * z
        else:
            x = mean

    return x.detach().cpu()

@torch.no_grad()
def ddim_sample(model, shape, n_steps=50, eta=0.0, device=None):
    T = 1000
    alpha_bar = ALPHA_BAR__T.to(device)  

    times = torch.linspace(0, T - 1, n_steps, device=device).long().flip(0)

    x = torch.randn(shape, device=device)

    B = shape[0]

    for i in range(n_steps):
        t = times[i].item()
        t_prev = times[i + 1].item() if i < n_steps - 1 else -1

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        a_t = alpha_bar[t]                          
        a_prev = torch.tensor(1.0, device=device) if t_prev < 0 else alpha_bar[t_prev]

        eps = model(x, t_batch)                   

      
        x0 = (x - torch.sqrt(1.0 - a_t) * eps) / (torch.sqrt(a_t) + 1e-8)
        x0 = x0.clamp(-1, 1)                       

        
        sigma = (
            eta
            * torch.sqrt((1.0 - a_prev) / (1.0 - a_t + 1e-8))
            * torch.sqrt(torch.clamp(1.0 - a_t / (a_prev + 1e-8), min=0.0))
        )

        c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0))

        z = torch.randn_like(x) if (eta > 0 and t_prev >= 0) else 0.0

        x = torch.sqrt(a_prev) * x0 + c * eps + sigma * z

    return x
