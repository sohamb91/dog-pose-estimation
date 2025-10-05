import os
from tqdm import tqdm
import torch

def train_model(model, criterion, train_dataloader, val_dataloader, lr_scheduler, optimizer,device = "cuda", num_epochs = 1):
    train_losses = []
    val_losses = []
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        # Training loop
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        for batch in progress_bar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            current_batch_size = batch["pixel_values"].shape[0]
            heatmaps_gt = batch.pop("heatmaps_gt")
            outputs = model(**batch)
            loss = criterion(outputs.heatmaps, heatmaps_gt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.5)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)
        print(f"Train loss Loss: {epoch_loss}")
        lr_scheduler.step(epoch_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                heatmaps_gt = batch.pop("heatmaps_gt")
                outputs = model(**batch)
                loss = criterion(outputs.heatmaps, heatmaps_gt)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")
        if (epoch + 1) % 5 == 0:
            save_path = f"./pose_estimattion_with_aug{epoch+1}"
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving model to {save_path}")
            model.save_pretrained(save_path)
    return train_losses, val_losses