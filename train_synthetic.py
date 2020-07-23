"""Main training script for synthetic problems."""

import argparse
import os
import time
import scipy.stats as st
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import higher

import layers
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder

OUTPUT_PATH = "./outputs/synthetic_outputs"


def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()
    metrics = {"train_loss": np.mean(qry_losses)}
    wandb.log(metrics, step=step_idx)
    meta_opt.step()


def test(step_idx, data, net, inner_opt_builder, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    wandb.log(test_metrics, step=step_idx)
    return avg_qry_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cpu")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    wandb.init(project="weight_sharing_toy", dir=OUTPUT_PATH)
    args = parser.parse_args()
    wandb.config.update(args)
    cfg = wandb.config
    device = torch.device(cfg.device)
    db = SyntheticLoader(device, problem=cfg.problem, k_spt=cfg.k_spt, k_qry=cfg.k_qry)

    if cfg.problem in ["2d_rot8_flip", "2d_rot8"]:
        c_o = 24 if cfg.problem == "2d_rot8" else 48
        if cfg.model == "share_conv":
            net = nn.Sequential(layers.ShareConv2d(1, c_o, 3, bias=False)).to(device)
        elif cfg.model == "conv":
            net = nn.Sequential(nn.Conv2d(1, c_o, 3, bias=False)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model}")
    elif cfg.problem in ["rank1", "rank2", "rank5"]:
        if cfg.model == "lc":
            net = nn.Sequential(layers.LocallyConnected1d(1, 1, 68, kernel_size=3, bias=False)).to(
                device
            )
        elif cfg.model == "fc":
            net = nn.Sequential(nn.Linear(70, 68, bias=False)).to(device)
        elif cfg.model == "conv":
            net = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, bias=False)).to(device)
        elif cfg.model == "share_fc":
            latent = {"rank1": 3, "rank2": 6, "rank5": 30}[cfg.problem]
            net = nn.Sequential(layers.ShareLinearFull(70, 68, bias=False, latent_size=latent)).to(
                device
            )
        else:
            raise ValueError(f"Invalid model {cfg.model}")

    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode
    )
    if cfg.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)

    start_time = time.time()
    for step_idx in range(cfg.num_outer_steps):
        data, _filters = db.next(32, "train")
        train(step_idx, data, net, inner_opt_builder, meta_opt, cfg.num_inner_steps)
        if step_idx == 0 or (step_idx + 1) % 100 == 0:
            test_data, _filters  = db.next(300, "test")
            val_loss = test(
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                cfg.num_inner_steps,
            )
            if step_idx > 0:
                steps_p_sec = (step_idx + 1) / (time.time() - start_time)
                wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
                print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}")


if __name__ == "__main__":
    main()
