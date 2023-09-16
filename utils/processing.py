import torch
from torch import tensor
from torchvision import transforms

import numpy as np
import PIL
import torch.nn.functional as F 
from skimage import morphology
from typing import Tuple
from PIL import ImageFilter
from sklearn import random_projection
from tqdm import tqdm

from data import mvtec_classes


backbones = {
    'WideResNet50':'wide_resnet50_2',
    'ResNet101':'RN101',
    'ResNet50':'RN50',
    'ResNet50-4':'RN50x4',
    'ResNet50-16':'RN50x16',
}

dataset_scale_factor = {
    'WideResNet50': 1,
    'ResNet101': 1,
    'ResNet50': 1,
    'ResNet50-4': 2,
    'ResNet50-16': 4,
}

def get_coreset(
        memory_bank: tensor,
        l: int = 1000,  # Coreset target
        eps: float = 0.09,
) -> tensor:
    """
        Returns l coreset indexes for given memory_bank.

        Args:
        - memory_bank:     Patchcore memory bank tensor
        - l:               Number of patches to select
        - eps:             Sparse Random Projector parameter

        Returns:
        - coreset indexes
    """

    coreset_idx = []  # Returned coreset indexes
    idx = 0

    # Fitting random projections
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        memory_bank = torch.tensor(transformer.fit_transform(memory_bank))
    except ValueError:
        print("Error: could not project vectors. Please increase `eps`.")

    # Coreset subsampling
    print(f'Start Coreset Subsampling...')

    last_item = memory_bank[idx: idx + 1]   # First patch selected = patch on top of memory bank
    coreset_idx.append(torch.tensor(idx))
    min_distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # Norm l2 of distances (tensor)

    # Use GPU if possible
    if torch.cuda.is_available():
        last_item = last_item.to("cuda")
        memory_bank = memory_bank.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(l - 1)):
        distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)    # L2 norm of distances (tensor)
        min_distances = torch.minimum(distances, min_distances)                         # Verical tensor of minimum norms
        idx = torch.argmax(min_distances)                                               # Index of maximum related to the minimum of norms

        last_item = memory_bank[idx: idx + 1]   # last_item = maximum patch just found
        min_distances[idx] = 0                  # Zeroing last_item distances
        coreset_idx.append(idx.to("cpu"))       # Save idx inside the coreset

    return torch.stack(coreset_idx)


def gaussian_blur(img: tensor) -> tensor:
    """
        Apply a gaussian smoothing with sigma = 4 over the input image.
    """
    # Setup
    blur_kernel = ImageFilter.GaussianBlur(radius=4)
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()

    # Smoothing
    max_value = img.max()   # Maximum value of all elements in the image tensor
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
    blurred_map = pil_to_tensor(blurred_pil) * max_value

    return blurred_map


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        
    return PIL.Image.fromarray(tensor)


def display_backbones():
    vanilla = True
    print("Vanilla PatchCore backbone:")
    print(f"- WideResNet50")
    print("CLIP Image Encoder architectures for PatchCore backbone:")
    for k, _ in backbones.items():
        if vanilla:
            vanilla = False
            continue
        print(f"- {k}")
    print()
    

def display_MVTec_classes():
    print(mvtec_classes())

def compute_mask(anomaly_map: np.array,
                 threshold: float = 200,
                 kernel_size: int = 4,
                 origin_img_size: Tuple[int] = (900, 900)
                 ):    
    anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)
    # anomaly_map = cv2.resize(anomaly_map[0], dsize=origin_img_size)

    mask = np.zeros_like(anomaly_map)
    mask[anomaly_map > threshold] = 1
    mask = mask[0]
    # mask = cv2.resize(mask, dsize=origin_img_size)

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)
    return mask

def extract_features(model,
                     image: torch.tensor,
                     output_size: int = 224):
    feature_maps = model.forward(image)
    avg = torch.nn.AvgPool2d(3, stride=1)
    fmap_size = feature_maps[0].shape[-2]         # Feature map sizes h, w
    resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
    resized_maps = [resize(avg(fmap)) for fmap in feature_maps]
    concated_feature = torch.cat(resized_maps, 1)
    resized_feature = F.interpolate(concated_feature,
                                    size=(output_size, output_size),
                                    mode="bilinear")
    return resized_feature
    