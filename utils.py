import math
import torch
import os
from torchvision.utils import make_grid, save_image
import shutil



def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    Args:
        timesteps: 1D tensor of shape [B] 
        embedding_dim: dimension of the embedding

    Returns:
        Tensor of shape [B, embedding_dim]
    """
    assert timesteps.dim() == 1  # [B]

    d_half = embedding_dim // 2

    # log(10000) / (d_1/2 - 1)
    emb_scale = math.log(10000) / (d_half - 1)

    # exp(-i * emb_scale)
    emb = torch.exp(torch.arange(d_half, device=timesteps.device, dtype=torch.float32)* -emb_scale)

    # timesteps[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :] #broadcast to each

    # concat sin and cos
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

@torch.no_grad()
def save_forward_noising_preview(X0, alpha_bar, out_path, timesteps=(1, 10, 50, 100, 250, 500, 999)):
    """
    X0: (B,C,H,W) in [-1,1] or [0,1] (doesn't matter for sanity)
    alpha_bar: (T+1,) tensor where alpha_bar[t] is cumulative product
    """
    device = X0.device
    B = X0.shape[0]
    previews = []
    for t in timesteps:
        t = int(t)
        eps = torch.randn_like(X0)
        a = alpha_bar[t].to(device)  # scalar
        xt = (a.sqrt() * X0) + ((1 - a).sqrt() * eps)
        previews.append(xt)

    grid = make_grid(torch.cat(previews, dim=0), nrow=B, normalize=True, value_range=(-1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)
    

# @torch.no_grad()
# def save_samples(model, sample_fn, out_path, n=10, shape=(3,32,32), device="cuda"):
#     model.eval()
#     C,H,W = shape
#     x = sample_fn(model, B=n, C=C, H=H, W=W, device=device)  # returns (n,C,H,W)
#     grid = make_grid(x, nrow=5, normalize=True, value_range=(-1, 1))
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     save_image(grid, out_path)
#     model.train()
    
@torch.no_grad()
def save_samples(
    model,
    sampling_fn,
    total=50_000,
    batch_size=256,
    steps=50,
    eta=0.0,
    image_size=32,
    device="cuda",
    fake_dir = "fid_fake",
    
):  
    shutil.rmtree(fake_dir, ignore_errors=True)  # delete if exists
    os.makedirs(fake_dir, exist_ok=True)         # recreate empty
    n_batches = math.ceil(total / batch_size)
    idx = 0

    model.eval()

    for b in range(n_batches):
        cur_bs = min(batch_size, total - idx)

        # Sample in [-1, 1] or whatever your sampler returns
        x = sampling_fn(
            model=model,
            shape=(cur_bs, 3, image_size, image_size),

            n_steps=steps,
            eta=eta,
            device=device,
        )

        # Convert to [0,1] for save_image
        x = (x.clamp(-1, 1) + 1) * 0.5

        for i in range(cur_bs):
            save_image(x[i], os.path.join(fake_dir, f"{idx:06d}.png"))
            idx += 1

        if (b + 1) % 10 == 0:
            print("Saved", idx, "images")