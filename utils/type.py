from typing import TypedDict, List
import numpy as np

class SplittedMaskInfo(TypedDict): 
    mask: np.array
    bbox: List[int]