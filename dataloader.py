import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self,path=None):
        # dataset path
        self.imgs_path = path
        # list all files
        self.data = glob.glob(self.imgs_path + "*")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
         
#         # resize (if required)
#         if img.shape[0] >= 256 and img.shape[1] >= 256:
#             interpolation_type = cv2.INTER_LINEAR
#             img = cv2.resize(img, self.img_dim,interpolation_type)
            
        img_tensor = torch.from_numpy(img)
        class_id = idx
        return img_tensor, class_id