import torch
from tqdm import tqdm


NUM_TIMESTEPS = 1000
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BETA__T = torch.linspace(10**-4, 0.02, NUM_TIMESTEPS, device=device)
ALPHA__T = 1.0 - BETA__T
ALPHA_BAR__T = torch.cumprod(ALPHA__T, dim = 0)





def forwardDiffusion(X0__BCHW):
    B,_,_,_ = X0__BCHW.shape
    eps__BCHW = torch.randn_like(X0__BCHW, device=X0__BCHW.device)
    t__B = torch.randint(low=0, high=NUM_TIMESTEPS, size=(B,), device=X0__BCHW.device)
    alpha_bar_root_t__B111 = torch.sqrt(ALPHA_BAR__T[t__B])[:, None, None, None]
    alpha_bar_1root_t__B111 = torch.sqrt(1 - ALPHA_BAR__T[t__B])[:, None, None, None]
    Xt_BCHW = alpha_bar_root_t__B111 * X0__BCHW + alpha_bar_1root_t__B111 * eps__BCHW
    return eps__BCHW, t__B, Xt_BCHW


#Normalize

def diffusionTrainStep(model, optimizer, loss_fn, X0):
    # sample a batch from the data set: Normalize to [-1,1]
    # sample a batch of t from Uniform({1,...,T})
    # sample a batch of noise from a standard normal distribution    
    # Already Normalized
    X0__BCHW = X0.to(device)
    eps__BCHW, t__B, Xt_BCHW = forwardDiffusion(X0__BCHW)
    eps_pred__BCHW = model(Xt_BCHW, t__B)
    loss = loss_fn(eps_pred__BCHW, eps__BCHW)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
        
    return loss.item()

 
@torch.no_grad()
def sample(model, B, C, H, W):
    model.eval()
    device = next(model.parameters()).device
    XT__BCHW = torch.randn((B,C,H,W), device = device)
    XT_copy__BCHW = XT__BCHW.clone()
    for t in range(NUM_TIMESTEPS -1, -1, -1):
        t__B = torch.full((B,), t, device=device, dtype=torch.long)
        
        if t>0: z = torch.randn_like(XT__BCHW)
        else: z = torch.zeros_like(XT__BCHW)
        
        eps_pred = model(XT__BCHW, t__B)

        alpha_t = ALPHA__T[t__B][:, None, None, None]          # (B,1,1,1)
        beta_t  = BETA__T[t__B][:, None, None, None]           # (B,1,1,1)
        ab_t    = ALPHA_BAR__T[t__B][:, None, None, None]      # (B,1,1,1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (XT__BCHW - ((1 - alpha_t) / torch.sqrt(1 - ab_t)) * eps_pred)

        XT__BCHW = mean + (torch.sqrt(beta_t) * z if t > 0 else 0.0)


    return XT__BCHW, XT_copy__BCHW
        
        
        
        

        
    
    