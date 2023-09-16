from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List 

from data import DEFAULT_RESIZE, DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD

class TrainDataset(Dataset): 
    def __init__(self,
                 data_path: List[str]
                 ):
        super().__init__()
        self.paths = data_path 
        self.transform = transforms.Compose([
            transforms.Resize(DEFAULT_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(DEFAULT_SIZE),  
            transforms.ToTensor(),  
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = self.transform(img)

        return img, [0]
    
class ValidDataset(Dataset): 
    def __init__(self,
                 data_path: List[str],
                 labels: List[int]
                 ):
        super().__init__()
        self.paths = data_path
        self.labels = labels 
        self.transform = transforms.Compose([
            transforms.Resize(DEFAULT_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(DEFAULT_SIZE),  
            transforms.ToTensor(),  
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        label = self.labels[index]
        img = self.transform(img)

        return img, [0], label


class InferenceDataset(Dataset):
    """
    Data Loader for inference image paths 
    # TODO Histogram Equalization function add in getitem function 
    """
    def __init__(self,
                 data_path: List[str]
                 ):
        super().__init__()
        self.paths = data_path
        self.transform = transforms.Compose([
            transforms.Resize(DEFAULT_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(DEFAULT_SIZE),  
            transforms.ToTensor(),  
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = self.transform(img)
        return img 