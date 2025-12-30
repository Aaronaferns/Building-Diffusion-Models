import math
import torch




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