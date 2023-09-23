import numpy as np
from data import DEFAULT_SIZE
from pathlib import Path 
from typing import List 
import cv2 

CLASS_COLORS = {
    1: (0, 0, 255),   # Red
    2: (0, 165, 255), # Orange 
    3: (0, 255, 255), # Yellow
    4: (0, 255, 0),   # Green
    5: (255, 0, 0),   # Blue
    6: (255, 0, 127), # Zazu
    7: (255, 0, 255)  # Purple
}


def visualize_cls_mask(origin_img_path: str, 
                       mask: np.array,
                       cls: int,
                       save_dir: str = "results/"
                       ):
    save_path = Path(save_dir) / (str(Path(origin_img_path).parent.name) + "_" + str(Path(origin_img_path).name))
    save_path.parent.mkdir(parents=True,
                           exist_ok=True)
    if save_path.exists():
        img = cv2.imread(str(save_path))
    else: 
        img = cv2.imread(origin_img_path)
    resized_img = cv2.resize(img, 
                             dsize=(DEFAULT_SIZE, DEFAULT_SIZE))
    
    cls_mask = mask * cls 
    output_mask = np.zeros((mask.shape[0], 
                            mask.shape[1], 3), 
                            dtype=np.uint8)
    
    for class_id in range(1, 8):
        class_mask = (cls_mask == class_id).astype(np.uint8)
        color = CLASS_COLORS[class_id]
        output_mask[class_mask == 1] = color

    displayed_img = cv2.addWeighted(output_mask, 0.4, np.array(resized_img), 0.6, 0)
    cv2.imwrite(str(save_path), displayed_img)