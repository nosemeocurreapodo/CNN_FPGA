import os
import gzip
import struct
import torch
from torch.utils.data import Dataset, DataLoader
import requests

DATA_DIR = './data/mnist_raw'
os.makedirs(DATA_DIR, exist_ok=True)

# URLs for MNIST files
urls = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

# Filenames
filenames = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}

# 1. Downloading data if not already present
for key in filenames:
    filepath = os.path.join(DATA_DIR, filenames[key])
    if not os.path.exists(filepath):
        print(f"Downloading {key}...")
        r = requests.get(urls[key], stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

def parse_idx_images(filepath):
    """Parse MNIST images from the given gzipped IDX file."""
    with gzip.open(filepath, 'rb') as f:
        # The first 16 bytes contain magic number, number of images, rows and cols
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the rest of the data as unsigned bytes and convert to float tensors
        data = f.read(num_images * rows * cols)
        images = torch.frombuffer(data, dtype=torch.uint8)
        images = images.view(num_images, 1, rows, cols).float() / 255.0
        return images

def parse_idx_labels(filepath):
    """Parse MNIST labels from the given gzipped IDX file."""
    with gzip.open(filepath, 'rb') as f:
        # The first 8 bytes contain magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        data = f.read(num_labels)
        labels = torch.frombuffer(data, dtype=torch.uint8).long()
        return labels

# Parse the data
train_images = parse_idx_images(os.path.join(DATA_DIR, filenames['train_images']))
train_labels = parse_idx_labels(os.path.join(DATA_DIR, filenames['train_labels']))
test_images = parse_idx_images(os.path.join(DATA_DIR, filenames['test_images']))
test_labels = parse_idx_labels(os.path.join(DATA_DIR, filenames['test_labels']))

# 2. Create a custom Dataset
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Example usage
train_dataset = MNISTDataset(train_images, train_labels)
test_dataset = MNISTDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Just to test loading a batch
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)  # Should print: torch.Size([64, 1, 28, 28]) torch.Size([64])
