import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np
import scipy.io as sio
import h5py
import cv2
import pickle

# Dataset class for NYU Depth V2 loaded from .mat file
class NYUMatDataset(Dataset):
    def __init__(self, images, depths, common_transform=None, image_transform=None, depth_transform=None):
        """
        images: np.array of shape [N, H, W, 3], float or uint8
        depths: np.array of shape [N, H, W], float indicating depth in meters
        transform: transform for images (PIL or tensor)
        depth_transform: transform for depths (tensor)
        """
        self.images = images
        self.depths = depths
        self.common_transform = common_transform
        self.image_transform = image_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image_np = np.transpose(self.images[idx], (0, 2, 1)).astype(np.float32)/255.0
        depth_np = np.expand_dims(np.transpose(self.depths[idx], (1, 0)), axis=0)
        
        # Convert to PIL Image for transforms
        #img = Image.fromarray(img_np.astype(np.uint8))
        #image_np = np.transpose(image_np, (2, 1, 0)) #Image.fromarray(image_np.astype(np.uint8))
        #depth_np = np.transpose(depth_np, (2, 1, 0))
        
        image = torch.from_numpy(image_np)
        depth = torch.from_numpy(depth_np)

        if self.image_transform is not None:
            image = self.image_transform(image)
            
        if self.depth_transform is not None:
            # depth_transform should be something that works on tensors
            depth = self.depth_transform(depth)
         
        data = torch.cat((image, depth), 0)   
        #data = torch.from_numpy(dataset_np)

        if self.common_transform is not None:
            data = self.common_transform(data)

        image = data[0:3,:,:]
        depth = data[3,:,:].unsqueeze(0)
        
        return image, depth


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to have the same size as x2
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class SimpleDepthEstimationNet(nn.Module):
    def __init__(self):
        super(SimpleDepthEstimationNet, self).__init__()
        # A very simple encoder-decoder style network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        #print("input shape: ", x.shape)
        x = self.encoder(x)
        #print("encoded shape: ", x.shape)
        x = self.decoder(x)
        #print("output shape: ", x.shape)
        return x


class UNetDepthEstimator(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetDepthEstimator, self).__init__()
        self.inc = DoubleConv(n_channels, 32) # 224
        self.down1 = Down(32, 64) #224 112
        self.down2 = Down(64, 128) #112 56
        self.down3 = Down(128, 256) #56 28
        #self.down4 = Down(128, 256) #14 7
        #self.down5 = Down(128, 256) #7
        
        #self.fc1 = nn.Linear(12544, 128)
        #self.fc2 = nn.Linear(128, 10)
        
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        #self.up4 = Up(32, 16)
        #self.up5 = Up(16, 8)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)    # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 64
        x4 = self.down3(x3) # 512
        #x5 = self.down4(x4) # 512
        #x6 = self.down5(x5) # 512
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        #x = self.up4(x, x1)
        #x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
    

class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, target):
        # pred, target: [N, 1, H, W]
        diff = torch.abs(target - pred)
        c = 0.2 * torch.max(diff).detach()
        mask1 = (diff <= c).float()
        mask2 = (diff > c).float()

        loss1 = diff * mask1
        loss2 = (diff**2 + c**2) / (2*c) * mask2
        loss = torch.mean(loss1 + loss2)
        return loss

    
def train_depth_estimator(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.MSELoss()  # MSE loss for regression
    criterion = BerHuLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, depths) in enumerate(train_loader):
            images = images.to(device)  # [N, 3, H, W]
            depths = depths.to(device)  # [N, 1, H, W]

            optimizer.zero_grad()
            preds = model(images)  # [N, 1, H, W]
            loss = criterion(preds, depths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, depths) in enumerate(val_loader):
                images = images.to(device)
                depths = depths.to(device)
                preds = model(images)
                loss = criterion(preds, depths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        torch.save(model.state_dict(), "depth.pt")
        
        state_dict = model.state_dict()
        state_dict_numpy = {}
        for key in state_dict:
            #print(f"{key}: {type(state_dict[key])}")
            state_dict_numpy[key] = state_dict[key].cpu().detach().numpy().tolist()
        #print(state_dict_numpy)
        #print(state_dict)
        
        with open("depth.pkl", "wb") as f:
            pickle.dump(state_dict_numpy, f)
            
        for images, depths, in val_loader:
            images = images.to(device)
            preds = model(images)

            depth = preds.cpu().detach().numpy()[0,0,:,:]
            
            #image = np.transpose(image, (2, 1, 0))
            
            depth = 255.0 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            cv2.imwrite(f'depth_{epoch}.png', depth)
            
            if epoch == 0:
                image = images.cpu().detach().numpy()[0,0,:,:]
                depth = depths.cpu().detach().numpy()[0,0,:,:]
                image = 255.0 * (image - np.min(image)) / (np.max(image) - np.min(image))
                depth = 255.0 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
                cv2.imwrite(f'image_ref.png', image)
                cv2.imwrite(f'depth_ref.png', depth)

            break


if __name__ == '__main__':
    # Load the .mat file
    # The file 'nyu_depth_data_labeled.mat' is commonly used for NYU v2
    # It typically contains 'images' and 'depths'
    #cwd = os.getcwd()
    mat_path = '/home/emanuel/workspace/CNN_FPGA/python/data/nyu_depth_v2/nyu_depth_v2_labeled.mat'
    #data = sio.loadmat(mat_path)
    # data['images'] -> shape: (H, W, 3, N), we need to transpose it to (N, H, W, 3)
    # data['depths'] -> shape: (H, W, N), transpose to (N, H, W)

    #images = data['images']  # shape (H, W, 3, N)
    #depths = data['depths']  # shape (H, W, N)

    # Transpose arrays to (N, H, W, C)
    # images: (H, W, 3, N) -> (N, H, W, 3)
    #images = np.transpose(images, (3, 0, 1, 2))
    # depths: (H, W, N) -> (N, H, W)
    #depths = np.transpose(depths, (2, 0, 1))
        
    # Load mat file data
    mat = h5py.File(mat_path, 'r', libver='latest', swmr=True)

    # Images are in 4D array (1449, 3, 640, 480), and depth maps are in 3D array (1449, 640, 480).
    # We can simply transpose the axes to get them in a format suitable for training.
    #images = np.transpose(mat["images"], (0, 3, 2, 1))#(0, 2, 3, 1))
    #depths = np.transpose(mat["depths"], (0, 2, 1))

    images = np.array(mat["images"])
    depths = np.array(mat["depths"])
    
    # Split into train/val (NYU depth commonly uses 795/654 split or similar)
    # For demonstration, let's do a simple split:
    num_samples = images.shape[0]
    train_count = int(0.8 * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    # Create dataset objects
    rgb_transform = None #T.Compose([
        #T.ToTensor(), 
        #T.Normalize(mean=[0.485, 0.456, 0.406],
        #            std=[0.229, 0.224, 0.225]),
        #T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),#, hue=(0.8, 1.2))
        #T.GaussianNoise(0.0, 0.1, True)
    #])

    depth_transform = T.Compose([T.Normalize(mean=[0.5], std=[0.2])])

    common_transform = T.Compose([T.RandomResizedCrop(size=(112, 112), antialias=True),
                                  T.RandomHorizontalFlip(p=0.5)])
    
    train_dataset = NYUMatDataset(images[train_indices], depths[train_indices],
                                  common_transform = common_transform, image_transform=rgb_transform, depth_transform=depth_transform)
    val_dataset = NYUMatDataset(images[val_indices], depths[val_indices],
                                common_transform = common_transform, image_transform=rgb_transform, depth_transform=depth_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    #model = SimpleDepthEstimationNet()
    model = UNetDepthEstimator()
    train_depth_estimator(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cuda')
