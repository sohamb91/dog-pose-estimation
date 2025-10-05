from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from mmpose.structures.bbox import (bbox_xywh2cs, bbox_cs2xywh)
from mmpose.datasets.transforms import (
    RandomFlip, RandomBBoxTransform, TopdownAffine
)
from mmpose.datasets.transforms.common_transforms import PhotometricDistortion
from mmpose.codecs.utils.gaussian_heatmap import (
    generate_unbiased_gaussian_heatmaps
)
from pathlib import Path
from mmcv.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

class AddGaussianNoise(BaseTransform):
    """Add Gaussian noise to keypoint coordinates (not image pixels)."""

    def __init__(self, mean=0.0, std=2.0):
        self.mean = mean
        self.std = std

    def transform(self, results):
        if 'keypoints' not in results:
            raise KeyError("Expected 'keypoints' in results dict")

        keypoints = results['keypoints']  # shape (num_instances, num_keypoints, 2)
        noise = np.random.normal(self.mean, self.std, keypoints.shape).astype(np.float32)
        results['keypoints'] = keypoints + noise
        return results

class VitPoseDataset(Dataset):
    def __init__(self, train_dir = None, processor = None,  num_keypoints = 24, augment=False):
        self.INPUT_SIZE = (192, 256)
        self.NUM_KPTS = 24
        self.train_dir = train_dir
        self.FEATURE_PER_KEYPOINT = 3
        self.processor = processor
        self.NUM_KEYPOINTS = num_keypoints
        self.augment = augment
        self.pipeline = None
        if self.augment:
            self.pipeline = Compose([
                RandomFlip(prob = 0.4, direction="horizontal"),
                RandomBBoxTransform(shift_factor = 0.15,
                 shift_prob = 0.3,
                 scale_factor= (0.7, 1.2),
                 scale_prob= 1.0,
                 rotate_factor = 25.0,
                 rotate_prob = 0.4),
                PhotometricDistortion(),
                AddGaussianNoise(mean = 0.0, std = 8.0)
            ]) 
        if self.train_dir is not None:
            self.train_images = sorted(list((Path(train_dir) / "images").glob("*.jpg")))
            self.train_labels = sorted(list((Path(train_dir) / "labels").glob("*.txt")))
    
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        im = self.train_images[idx]
        annotation_file = [im_file for im_file in self.train_labels if im_file.stem == im.stem][0]
        with open(annotation_file, "r") as f:
            contents = f.read()
            ann = [float(part) for part in contents.split(" ")]
        image = Image.open(im).convert('RGB')
        image_np = np.asarray(image)

        kpts = ann[5:]
        bbox = ann[1:5]
        im_height, im_width, _ = image_np.shape
        ##bbox
        x, y, w, h = bbox ## in yolo format
        x_coco = (x - w / 2)
        y_coco = (y - h / 2)
        bbox_denorm = [x_coco * im_width, y_coco * im_height, w * im_width, h * im_height]
        ##keypoints
        kpt_reshaped = torch.tensor(kpts, dtype=torch.float32).view(self.NUM_KPTS, -1)
        dim_tensor = torch.tensor((im_width, im_height)).expand(self.NUM_KPTS, 2)
        kpt_coords = kpt_reshaped[:, :2]
        kpt_vis = kpt_reshaped[:, 2:]
        kpt_denorm = kpt_coords * dim_tensor
        kpt_denorm = torch.cat((kpt_denorm, kpt_vis), dim=1)
        center, scale = bbox_xywh2cs(np.array(bbox_denorm), padding=1.25)
        results = {
            'img': image_np,
            'img_shape': (image_np.shape[0], image_np.shape[1]),
            'keypoints': kpt_denorm[:, :2].cpu().numpy().reshape( 1, -1,  2),
            'keypoints_visible': kpt_denorm[:, 2:].cpu().numpy().reshape(1,-1, 1),
            'bbox_center': np.array(center).reshape(1, -1),
            'bbox_scale': np.array(scale).reshape(1, -1),
            'flip_indices': [
                    6, 7, 8,        
                    9, 10, 11,      
                    0, 1, 2,        
                    3, 4, 5,        
                    12, 13,         
                    15, 14,         
                    16, 17,         
                    19, 18,         
                    20, 21, 22, 23  
                ]
        }
        if self.augment and self.pipeline is not None:
            results = self.pipeline(results)
        if self.augment:
            bbox_denorm = bbox_cs2xywh(results['bbox_center'], results['bbox_scale'], padding = 1.25).reshape(4)
        
        results_copy = results.copy()
        affine = TopdownAffine(input_size = self.INPUT_SIZE)
        transformed = affine(results_copy)
        STRIDE = 4
        K = transformed["transformed_keypoints"].reshape(-1,2)
        K_T = K[np.newaxis, ...] / STRIDE
        K_V  = transformed["keypoints_visible"].squeeze()
        K_V_T = K_V[np.newaxis, ...]
        h_w = self.INPUT_SIZE[0] // STRIDE
        h_h = self.INPUT_SIZE[1] // STRIDE
        heatmaps = generate_unbiased_gaussian_heatmaps(heatmap_size = (h_w, h_h), keypoints = K_T, keypoints_visible = K_V_T, sigma = 2.0)
        if self.processor:
            processed = self.processor(Image.fromarray(results["img"]), boxes=[[bbox_denorm]], return_tensors = "pt")
            results = {**processed}
        results["heatmaps_gt"] = heatmaps[0]
        return results
