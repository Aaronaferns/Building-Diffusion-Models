



from itertools import cycle
from diffusion import diffusionTrainStep, sample
import tqdm
from utils import save_samples
from cleanfid import fid

import time, torch, os
import matplotlib.pyplot as plt
import torch.nn.functional as F








def save_ckpt(model, ema, optimizer, step, loss_list, ckpt_dir, fid_list, best_fid = None,  fid_value = None):

    os.makedirs(ckpt_dir, exist_ok=True)

    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best_fid.pt")

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "optim": optimizer.state_dict(),
        "fid_list": fid_list,
        "losses": loss_list,
        "best_fid" :best_fid,
    }
    torch.save(ckpt, last_path)
    if fid_value is not None and best_fid is not None and fid_value < best_fid:
        best_fid = fid_value
        ckpt["best_fid"] = best_fid
        torch.save(ckpt, best_path)
        print("Saved new best_fid:", best_fid)

    return best_fid

def train(
    main_dir,
    start_step,
    real_dir,
    sample_dir,
    model, ema, dataLoader,
    optimizer, loss_fn,
    exp_no, 
    sampling_fn,
    device,
    soft_N,
    hard_N,
    ckpt_dir,
    save_dir,
    fid_warmup_steps=5000,
    fid_interval_N = 50000,
    fid_interval_n = 10000,
    loss_list = [],
    fid_list = [],
    best_fid = float('inf')
):
    
    #directories
    ckpt_dir1 = main_dir + f"/{exp_no}/" + ckpt_dir 
    save_dir1 = main_dir + f"/{exp_no}/" + save_dir 
    fdir1 = main_dir +"/" + real_dir 
    fdir2 = main_dir + f"/{exp_no}/" + sample_dir 
    
    
    
    model.train()
    data_iter = cycle(dataLoader)
    pbar = tqdm.tqdm(total=None, desc="Training", dynamic_ncols=True)

    loss_list = loss_list
    fid_list = fid_list
    step = start_step
    best_fid = best_fid
    
    try:
        while True:
            X0, _ = next(data_iter)

            loss = diffusionTrainStep(model, ema, optimizer, loss_fn, X0, device)
            loss_list.append(float(loss))

        
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}", step=step)

            

        
            do_fid = (step >= fid_warmup_steps) and (step % fid_interval_n == 0)

            if do_fid:
            
                t0 = time.time()

                if step % fid_interval_N == 0:
                    save_samples(
                                ema.ema,
                                sampling_fn,
                                total=hard_N,
                                batch_size=256,
                                steps=50,
                                eta=0.0,
                                image_size=32,
                                device=device,
                                fake_dir = fdir2,
                                    
                                )
                    fid_score = fid.compute_fid(fdir1+"/real_hard", fdir2, device=device,
                                        num_workers=0,     
                                        batch_size=64,      
                                        use_dataparallel=False,)

                elif step % fid_interval_n == 0:
                    save_samples(
                                ema.ema,
                                sampling_fn,
                                total=soft_N,
                                batch_size=256,
                                steps=50,
                                eta=0.0,
                                image_size=32,
                                device=device,
                                fake_dir = fdir2,
                                    
                                )
                    fid_score = fid.compute_fid(fdir1+"/real_soft", fdir2, device=device,
                                                    num_workers=0,     
                                                    batch_size=64,      
                                                    use_dataparallel=False,)
            
                    

                dt = time.time() - t0
                fid_list.append((step, float(fid_score)))
                pbar.write(f"[FID] step={step}  fid={fid_score:.4f}  ({dt:.1f}s)")
                pbar.set_postfix(loss=f"{loss:.4f}", step=step, fid=f"{fid_score:.3f}")
                best_fid = save_ckpt(model, ema, optimizer, step,loss_list, ckpt_dir1, fid_list = fid_list, best_fid=best_fid, fid_value=fid_score )
                save_loss_fid_plots(
                    ckpt_dir1,
                    save_dir1,
                )

            step += 1
            
    except KeyboardInterrupt:
        pbar.write("\n[INFO] Training interrupted. Saving latest checkpoint...")
        save_ckpt(model, ema, optimizer, step, loss_list, ckpt_dir1, fid_list = fid_list)
    finally:
        pbar.close()
        
        



def evaluate_fid(
    ema,
    sampling_fn,
    hard_N,
    fdir1,
    fdir2,
    device,
    batch_size = 256,
    steps = 50,
    eta = 0.0,
    image_size = 32,
    ):
    t0 = time.time()
    save_samples(
                ema.ema,
                sampling_fn = sampling_fn,
                total=hard_N,
                batch_size=batch_size,
                steps=steps,
                eta=eta,
                image_size=image_size,
                device=device,
                fake_dir = fdir2,
                    
                )
    fid_score = fid.compute_fid(
                            fdir1+"/real_hard", 
                            fdir2, device=device,
                            num_workers=0,     
                            batch_size=64,      
                            use_dataparallel=False,
                            )
    dt = time.time() - t0

    return fid_score, dt
    
def save_loss_fid_plots(
                checkpoint_path,
                results_save_path,
                ):
    
        
    os.makedirs(results_save_path, exist_ok=True)
    loss_plot_path = os.path.join(results_save_path, "loss_plot.png")
    fid_plot_path = os.path.join(results_save_path, "fid_plot.png")

    #delete any old figures
    for path in [loss_plot_path, fid_plot_path]:
        if os.path.exists(path):
            os.remove(path)
            
    ckpt = torch.load(checkpoint_path+"/last.pt", map_location="cpu", weights_only=False)

    best_fid = ckpt.get("best_fid")
    loss_list = ckpt.get("losses")
    fid_list = ckpt.get("fid_list")
    
    if best_fid is None: 
        print("best fid is None. Setting to default 500")
        best_fid = 500
    if loss_list:
        # ---- Loss plot ----
        plt.figure()
        plt.plot(loss_list)
        plt.xlabel("Training step")
        plt.ylabel("MSE loss")
        plt.title(f"DDPM Training Loss (best FID: {best_fid:.4f})")
        plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved loss plot to: {loss_plot_path}")

    if fid_list:
        steps, fids = zip(*fid_list) 

        plt.figure()
        plt.plot(steps, fids, marker="o")
        plt.xlabel("Training steps")
        plt.ylabel("FID")
        plt.title(f"Fréchet Inception Distance (Best FID: {best_fid:.4f})")
        plt.savefig(fid_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved fid plot to: {fid_plot_path}")
    
    

def model_load_latest_state(*, exp_no, main_dir, ckpt_dir, model, optimizer, ema, device):
    ckpt_path = main_dir + f"/{exp_no}/" + ckpt_dir + "/last.pt"
    print(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    ema.ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optim"])

    start_step = ckpt.get("step", 0)
    best_fid = ckpt.get("best_fid", float("inf"))
    loss_list = ckpt.get("losses", [])
    fid_list = ckpt.get("fid_list", [])
    if best_fid is None:
        best_fid = float("inf")
    if fid_list is None:
        fid_list = []
    if loss_list is None:
        loss_list = []
    return start_step, best_fid, loss_list, fid_list