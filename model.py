import torch
from torch import nn
from transformers import VitPoseForPoseEstimation

class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints = 24, model_ckpt = "usyd-community/vitpose-base-simple", fine_tune = True):
        super(self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_fine_tune = fine_tune
        self.model = VitPoseForPoseEstimation.from_pretrained(str(model_ckpt), device_map=self.device, attn_implementation="sdpa")
        self.num_keypoints = num_keypoints
    def setup_classifier_head(self):
        if self.is_fine_tune:
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'conv'):
                original_conv = self.model.head.conv
                new_conv = torch.nn.Conv2d(
                    original_conv.in_channels,
                    self.num_keypoints,  # Number of dog keypoints
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding
                )
                with torch.no_grad():
                    num_channels_to_copy = min(17, self.num_keypoints)
                    new_conv.weight[:num_channels_to_copy] = original_conv.weight[:num_channels_to_copy]
                    if original_conv.bias is not None:
                        new_conv.bias[:num_channels_to_copy] = original_conv.bias[:num_channels_to_copy]
                    nn.init.xavier_uniform_(new_conv.weight[17:])
                    if new_conv.bias is not None:
                        nn.init.constant_(new_conv.bias[17:], 0)
            self.model.head.conv = new_conv
    def forward(self, x): 
        return self.model(x)