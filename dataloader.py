from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 16
def collate_fn(examples):
    batch = {}
    d_keys = examples[0].keys()
    for key in examples[0].keys():
        if key == "heatmaps_gt":
            batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples])
        elif key == "pixel_values":
            batch[key] = torch.cat([ex[key] for ex in examples], dim = 0)
    return batch


# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory = True, num_workers = 4)
# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory = True, num_workers = 4)