from typing import List
import torch
import numpy as np
import pytorch_lightning as pl


class EntryDetector():
    def __init__(self, model: pl.LightningModule, width: int=28, height: int=28, characters: str="123456789ABCDEFG") -> None:
        
        super().__init__()
        
        self.width = width
        self.height = height
        self.characters = characters
        
        self.model = model
        self.model.eval()
    
    def get_entries(self, boxes: List[np.ndarray], n: int=9) -> str:
        boxes = np.array(boxes)
        boxes = torch.tensor(boxes.reshape(boxes.shape[0], 1, boxes.shape[1], boxes.shape[2]))
        result = self.model(boxes)
        return [self.characters[i] for i in torch.argmax(result[:, :n], dim=1)]

