"""Generate synthetic data for experiments."""

import argparse
import os
from e2cnn import gspaces
from e2cnn import nn as gnn
from scipy.special import softmax
import numpy as np
import torch
from torch import nn
from layers import LocallyConnected1d


def generate_1d(out_path):
    lc_layer = LocallyConnected1d(1, 1, 68, bias=False)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.randn(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        lc_layer.weight = nn.Parameter(torch.from_numpy(filt))
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 70).astype(np.float32)
        result = lc_layer(torch.from_numpy(inp))  # (20, 1, 68)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_1d_low_rank(out_path, rank=2):
    lc_layer = LocallyConnected1d(1, 1, 68, bias=False)
    xs, ys, ws = [], [], []
    connectivity = softmax(np.random.randn(68, rank), axis=1)  # shape == (68, rank)
    for task_idx in range(10000):
        basis = np.random.randn(rank, 3)
        filt = np.dot(connectivity, basis)  # shape == (68, 3)
        filt = np.reshape(filt, (1, 1, 1, 68, 3)).astype(np.float32)
        ws.append(filt)
        lc_layer.weight = nn.Parameter(torch.from_numpy(filt))
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 70).astype(np.float32)
        result = lc_layer(torch.from_numpy(inp))  # (20, 1, 68)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_2d_rot8(out_path):
    r2_act = gspaces.Rot2dOnR2(N=8)
    feat_type_in = gnn.FieldType(r2_act, [r2_act.trivial_repr])
    feat_type_out = gnn.FieldType(r2_act, 3 * [r2_act.regular_repr])
    conv = gnn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, bias=False)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        gnn.init.generalized_he_init(conv.weights, conv.basisexpansion)
        inp = gnn.GeometricTensor(torch.randn(20, 1, 32, 32), feat_type_in)
        result = conv(inp).tensor.detach().cpu().numpy()
        xs.append(inp.tensor.detach().cpu().numpy())
        ys.append(result)
        ws.append(conv.weights.detach().cpu().numpy())
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_2d_rot8_flip(out_path):
    r2_act = gspaces.FlipRot2dOnR2(N=8)
    feat_type_in = gnn.FieldType(r2_act, [r2_act.trivial_repr])
    feat_type_out = gnn.FieldType(r2_act, 3 * [r2_act.regular_repr])
    xs, ys, ws = [], [], []
    device = torch.device("cuda")
    conv = gnn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, bias=False).to(device)
    for task_idx in range(2000):
        gnn.init.generalized_he_init(conv.weights, conv.basisexpansion)
        inp = gnn.GeometricTensor(torch.randn(20, 1, 32, 32).to(device), feat_type_in)
        result = conv(inp).tensor.detach().cpu().numpy()
        xs.append(inp.tensor.detach().cpu().numpy())
        ys.append(result)
        ws.append(conv.weights.detach().cpu().numpy())
        del inp, result
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


TYPE_2_PATH = {
    "rank1": "./data/rank1.npz",
    "rank2": "./data/rank2.npz",
    "rank5": "./data/rank5.npz",
    "2d_rot8": "./data/2d_rot8.npz",
    "2d_rot8_flip": "./data/2d_rot8_flip.npz",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="rank1")
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]
    if os.path.exists(out_path):
        raise ValueError(f"File exists at {out_path}.")
    if args.problem == "rank1":
        generate_1d(out_path)
    elif args.problem == "rank2":
        generate_1d_low_rank(out_path, rank=2)
    elif args.problem == "rank5":
        generate_1d_low_rank(out_path, rank=5)
    elif args.problem == "2d_rot8":
        generate_2d_rot8(out_path)
    elif args.problem == "2d_rot8_flip":
        generate_2d_rot8_flip(out_path)
    else:
        raise ValueError(f"Unrecognized problem {args.problem}")


if __name__ == "__main__":
    main()

