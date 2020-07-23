"""Meta dataloader for synthetic problems."""

import numpy as np
import torch
from generate_synthetic_data import TYPE_2_PATH


class SyntheticLoader:
    def __init__(self, device, problem="default", k_spt=1, k_qry=19):
        self.device = device
        data = np.load(TYPE_2_PATH[problem])
        self.xs, self.ys, self.ws = data["x"], data["y"], data["w"]
        # xs shape: (10000, 20, c_i, ...)
        # ys shape: (10000, 20, c_o, ...)
        self.c_i, self.c_o = self.xs.shape[2], self.ys.shape[2]
        self.k_spt, self.k_qry = k_spt, k_qry
        assert k_spt + k_qry <= 20, "Max 20 k_spt + k_20"
        train_cutoff = int(0.8 * self.xs.shape[0])
        self.train_range = range(train_cutoff)
        self.test_range = range(train_cutoff, self.xs.shape[0])

    def next(self, n_tasks, mode="train"):
        rnge = self.train_range if mode == "train" else self.test_range
        task_idcs = np.random.choice(rnge, n_tasks, replace=False)
        xs, ys, ws = self.xs[task_idcs], self.ys[task_idcs], self.ws[task_idcs]
        num_examples = xs.shape[1]
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i in range(n_tasks):
            example_idcs = np.random.choice(num_examples, self.k_spt + self.k_qry, replace=False)
            spt_idcs, qry_idcs = example_idcs[: self.k_spt], example_idcs[self.k_spt :]
            x_spt.append(xs[i][spt_idcs])
            y_spt.append(ys[i][spt_idcs])
            x_qry.append(xs[i][qry_idcs])
            y_qry.append(ys[i][qry_idcs])
        x_spt = np.stack(x_spt)
        y_spt = np.stack(y_spt)
        x_qry = np.stack(x_qry)
        y_qry = np.stack(y_qry)
        data = [x_spt, y_spt, x_qry, y_qry]
        data = [torch.from_numpy(x.astype(np.float32)).to(self.device) for x in data]
        return data, ws
