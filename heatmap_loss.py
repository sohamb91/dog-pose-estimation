import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
    def forward(self, pred_heatmaps, gt_heatmaps, weights=None):
        B, K, H, W = pred_heatmaps.shape
        pred = pred_heatmaps.view(B, K, -1)
        gt   = gt_heatmaps.view(B, K, -1)
        loss = 0.0
        for j in range(K):
            pj = pred[:, j, :]   # (B, H*W)
            gj = gt[:, j, :]
            if weights is not None:
                wj = weights[:, j].view(B, 1)
                loss_j = self.criterion(pj * wj, gj * wj)
            else:
                loss_j = self.criterion(pj, gj)
            loss += loss_j
        return loss / K