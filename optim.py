
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

def get_parameter_groups(
    model,
    lr_backbone=3e-6,
    lr_head=1e-4,
    llrd_decay=0.75,
    wd_backbone=5e-3,
    wd_head=1e-3,
):
    parameter_groups = []
    num_layers = model.backbone.config.num_hidden_layers

    def get_layer_id(name):
        if "encoder.layer" in name:
            return int(name.split("encoder.layer.")[1].split(".")[0])
        elif "embedding" in name:
            return -1  # frozen anyway
        elif "encoder" in name or "backbone" in name:
            return 0  # non-layer-specific backbone parts
        return None

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "embedding" in name:
            param.requires_grad = False
            continue

        is_bias_or_norm = name.endswith(".bias") or any(norm in name for norm in ["ln", "bn", "norm"])

        # === Head ===
        if "head" in name:
            lr = lr_head
            wd = 0.0 if is_bias_or_norm else wd_head
            parameter_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

        # === Backbone ===
        elif "backbone" in name:
            layer_id = get_layer_id(name)
            if layer_id is None:
                continue
            # lr = lr_backbone * (llrd_decay ** (num_layers - layer_id))
            lr = lr_backbone * (llrd_decay ** (num_layers - layer_id - 1))
            wd = 0.0 if is_bias_or_norm else wd_backbone * (llrd_decay ** (num_layers - layer_id))
            parameter_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

    return parameter_groups



def get_params_and_optim(model):
    param_groups = get_parameter_groups(
        model,
        lr_backbone=4e-6,
        lr_head=1e-4,
        llrd_decay=0.75,
        wd_backbone=1e-2,
        wd_head=1e-2
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-8
    )
    return optimizer, lr_scheduler
