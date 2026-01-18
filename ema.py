import copy
import torch

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device=None):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int | None = None, warmup: bool = True):
        if warmup and step is not None:
            decay = min(self.decay, 1.0 - 1.0 / (step + 1))
        else:
            decay = self.decay

        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k, v in esd.items():
            if k not in msd:
                continue
            model_v = msd[k]
            if not torch.is_floating_point(model_v):
                esd[k].copy_(model_v)
            else:
                v.mul_(decay).add_(model_v, alpha=1.0 - decay)

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.ema.load_state_dict(state["ema"], strict=True)
