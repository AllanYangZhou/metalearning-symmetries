"""Custom inner loop optimizers for use with higher."""
import collections
import torch
from torch.optim import SGD, Adam


def is_warp_layer(name):
    return "warp" in name


NAME_TO_INNER_OPT_CLS = {
    "maml": SGD,
    "maml_adam": Adam,
}


# TODO(allanz): Refactor into a module (or several), similar to ebn in higher/examples.
class InnerOptBuilder:
    def __init__(self, network, device, opt_name, init_lr, init_mode, lr_mode, ext_metaparams=None):
        self.network = network
        self.opt_name = opt_name
        self.init_lr = init_lr
        self.init_mode = init_mode
        self.lr_mode = lr_mode
        # metaparams that are not neural network params (e.g., learned lrs)
        if ext_metaparams:
            self.ext_metaparams = ext_metaparams
        else:
            self.ext_metaparams = self.make_ext_metaparams(device)
        self.inner_opt_cls = NAME_TO_INNER_OPT_CLS[opt_name]
        self.inner_opt = NAME_TO_INNER_OPT_CLS[opt_name](self.param_groups, lr=self.init_lr)

    def make_ext_metaparams(self, device):
        ext_metaparams = {}
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            if self.lr_mode == "per_layer":
                inner_lr = torch.tensor(self.init_lr).to(device)
                inner_lr.requires_grad = True
                ext_metaparams[f"{name}_lr"] = inner_lr
            elif self.lr_mode == "per_param":
                inner_lr = self.init_lr * torch.ones_like(param).to(device)
                inner_lr.requires_grad = True
                ext_metaparams[f"{name}_lr"] = inner_lr
            elif self.lr_mode == "fixed":
                pass
            else:
                raise ValueError(f"Unrecognized lr_mode: {self.lr_mode}")
        return ext_metaparams

    @property
    def metaparams(self):
        metaparams = {}
        metaparams.update(self.ext_metaparams)
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or self.init_mode == "learned":
                metaparams[name] = param
        return metaparams

    @property
    def param_groups(self):
        param_groups = []
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            param_groups.append({"params": param})
        return param_groups

    @property
    def overrides(self):
        overrides = collections.defaultdict(list)
        for name, param in self.network.named_parameters():
            if is_warp_layer(name) or not param.requires_grad:
                # Ignore symmetry params in the inner loop.
                continue
            if self.lr_mode == "per_layer":
                overrides["lr"].append(self.ext_metaparams[f"{name}_lr"])
            elif self.lr_mode == "per_param":
                overrides["lr"].append(self.ext_metaparams[f"{name}_lr"])
            elif self.lr_mode == "fixed":
                pass
            else:
                raise ValueError(f"Unrecognized lr_mode: {self.lr_mode}")
        return overrides

