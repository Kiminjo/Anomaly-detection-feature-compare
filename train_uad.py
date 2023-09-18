from data import  TrainDataset, ValidDataset, DEFAULT_SIZE
from models import PatchCore
from utils import backbones, dataset_scale_factor

from torch.utils.data import DataLoader
from pathlib import Path
import warnings 
import random 

warnings.filterwarnings(action="ignore")
random.seed(21)

def patchcore_train(train_loader: DataLoader,
                    valid_loader: DataLoader,
                    backbone: str = 'WideResNet50',
                    checkpoint_path: str = "checkpoints/weight.pt",
                    f_coreset: float = 0.1
                    ) -> None:

    # Vanilla or Clip version
    vanilla = backbone == "WideResNet50"

    results = {}    # Image level ROC-AUC, Pixel level ROC-AUC
    if vanilla:
        size = DEFAULT_SIZE
    elif backbone == 'ResNet50':    # RN50
        size = 224
    elif backbone == 'ResNet50-4':  # RN50x4
        size = 288
    elif backbone == 'ResNet50-16': # RN50x16
        size = 384
    elif backbone == 'ResNet101':   # RN50x101
        size = 224
    else:
        raise Exception('You can use the following nets: ResNet50, ResNet50-4, ResNet50-16, ResNet50-64, ResNet101')

    print(f'Running PatchCore...')
    patch_core = PatchCore(f_coreset, 
                           vanilla=vanilla, 
                           backbone=backbones[backbone], 
                           image_size=size)            
    print(f'Training...')
    patch_core.fit(train_loader, scale=dataset_scale_factor[backbone])

    print(f'Testing...')
    image_rocauc, pixel_rocauc = patch_core.evaluate(valid_loader)

    print(f'Results:')
    results["image_rocauc"] = float(image_rocauc) 
    results["pixel_rocauc"] = float(pixel_rocauc)
    print(f'- Image-level ROC AUC = {image_rocauc:.3f}')
    print(f'- Pixel-level ROC AUC = {pixel_rocauc:.3f}\n')

    patch_core.save(save_path=checkpoint_path)


if __name__=='__main__':
    checkpoint = "checkpoints/weight.pt"
    data_dir = Path("datasets/bottle/")
    defects = ["good", "broken_large", "broken_small", "contamination"]

    # Componet 0. Get Train and Valid Loader 
    train_data_paths = [str(p) for p in (data_dir / "train/good").glob("*.png")]
    valid_data_paths, valid_labels = [], []
    for defect_idx, defect in enumerate(defects): 
        img_paths = [str(p) for p in (data_dir / "test" / defect).glob("*.png")]
        labels = [defect_idx] * len(img_paths)

        valid_data_paths += img_paths
        valid_labels += labels 
    patchcore_labels = [1 if l !=0 else 0 for l in valid_labels]

    trainset = TrainDataset(data_path=train_data_paths)
    validset = ValidDataset(data_path=valid_data_paths,
                            labels=patchcore_labels)
    
    train_loader = DataLoader(trainset,
                              num_workers=8,
                              shuffle=True)
    valid_loader = DataLoader(validset,
                              num_workers=8,
                              shuffle=False)
    
    # Component 1. Train PatchCore Model 
    checkpoint = "checkpoints/split_pipeline.pt"
    patchcore_train(train_loader=train_loader,
                    valid_loader=valid_loader,
                    checkpoint_path=checkpoint)