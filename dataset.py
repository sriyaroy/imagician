import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import os

# downsample and crop to 64 x 64 to create low_res counterpart - only run once when new data has been added
def lowRes(image_path, target_path):
    hr_image = cv2.imread(image_path)
    height, width = hr_image.shape[:2]

    if height > 128:
        hr_image = hr_image[:, 0:128]
    if width > 128:
        hr_image = hr_image[0:128, :]

    lr_dim = (int(width * 0.5), int(height * 0.5))
    lr_image = cv2.resize(hr_image, lr_dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite(image_path, hr_image)
    cv2.imwrite(target_path, lr_image)

lr_dir = '/Users/sriyaroy/Developer/Super-resolution/Datasets/data-64x64-128x128/low_res'
hr_dir = '/Users/sriyaroy/Developer/Super-resolution/Datasets/data-64x64-128x128/high_res'

# Create dataset object for custom paired dataset
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        # List of all filenames
        self.filenames = os.listdir(self.lr_dir)
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get the file name
        filename = self.filenames[idx]

        # Build the full path for LR and HR images
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        # Load the images
        lr_im = cv2.cvtColor(cv2.imread(lr_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        hr_im = cv2.cvtColor(cv2.imread(hr_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Apply the transforms
        if self.transform_lr:
            lr_im = self.transform_lr(lr_im)
        if self.transform_hr:
            hr_im = self.transform_hr(hr_im)

        # return the images
        return lr_im, hr_im
    
## Use your custom data loader and apply transforms required
transform_lr = transforms.Compose([transforms.ToTensor()])
transform_hr = transforms.Compose([transforms.ToTensor()])

dataset = PairedImageDataset(lr_dir, hr_dir, transform_lr, transform_hr)

# Split train/test
train_ratio = 0.7
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(25)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

'''# USE TO CHECK WHICH FILES ARE IN TEST DATASET
def list_files(msg, subset, n=None):
    names = [subset.dataset.filenames[i] for i in subset.indices]
    print(f'{msg}: {len(names)} files')
    for name in names[:n]:
        print(' ', name)

list_files('Test files:', test_dataset, n=10) 
'''