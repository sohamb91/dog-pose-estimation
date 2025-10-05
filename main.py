import torch
from dataset import VitPoseDataset
from pathlib import Path
from dataloader import collate_fn
from torch.utils.data import DataLoader
from model import PoseEstimationModel
from heatmap_loss import HeatmapMSELoss
from optim import get_params_and_optim
from train import train_model
from transformers import AutoProcessor

train_dir = (Path.cwd() / "dog-pose_dataset" / "train")
val_dir = (Path.cwd() / "dog-pose_dataset" / "val")

BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple", use_fast=False)
train_dataset = VitPoseDataset(train_dir = train_dir, processor=processor, augment = True)
val_dataset = VitPoseDataset(train_dir = val_dir, processor=processor)        

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory = True, num_workers = 4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory = True, num_workers = 4)

model = PoseEstimationModel().to(device)

# print(model.model.config)


criterion = HeatmapMSELoss()
optimizer, scheduler = get_params_and_optim(model)

if __name__ == "__main__":
    train_model(model, criterion, train_dataloader, val_dataloader, scheduler, optimizer, num_epochs=100)
