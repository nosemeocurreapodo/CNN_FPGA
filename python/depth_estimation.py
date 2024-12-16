import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import scipy.io as sio
import h5py
import cv2

# Dataset class for NYU Depth V2 loaded from .mat file
class NYUMatDataset(Dataset):
    def __init__(self, images, depths, transform=None, depth_transform=None):
        """
        images: np.array of shape [N, H, W, 3], float or uint8
        depths: np.array of shape [N, H, W], float indicating depth in meters
        transform: transform for images (PIL or tensor)
        depth_transform: transform for depths (tensor)
        """
        self.images = images
        self.depths = depths
        self.transform = transform
        self.depth_transform = depth_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img_np = self.images[idx]  # shape: (H, W, 3)
        depth_np = self.depths[idx]  # shape: (H, W)

        # Convert to PIL Image for transforms
        img = Image.fromarray(img_np.astype(np.uint8))
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        if self.transform is not None:
            img = self.transform(img)

        if self.depth_transform is not None:
            # depth_transform should be something that works on tensors
            depth_tensor = self.depth_transform(depth_tensor)

        return img, depth_tensor


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
        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)

        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.up4 = Up(16, 8)
        self.outc = nn.Conv2d(8, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)    # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 64
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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
        for images, depths, in val_loader:
            images = images.to(device)
            preds = model(images)

            depth = preds.cpu().detach().numpy()[0,0,:,:]
            
            #image = np.transpose(image, (2, 1, 0))
            
            depth = 255.0 * depth / np.max(depth)
            cv2.imwrite(f'depth_{epoch}.png', depth)
            
            if epoch == 0:
                image = images.cpu().detach().numpy()[0,0,:,:]
                depth = depths.cpu().detach().numpy()[0,0,:,:]
                image = 255.0 * image / np.max(image)
                depth = 255.0 * depth / np.max(depth)
                cv2.imwrite(f'image_ref.png', image)
                cv2.imwrite(f'depth_ref.png', depth)

            break


if __name__ == '__main__':
    # Load the .mat file
    # The file 'nyu_depth_data_labeled.mat' is commonly used for NYU v2
    # It typically contains 'images' and 'depths'
    cwd = os.getcwd()
    mat_path = cwd + '/data/nyu_depth_v2/nyu_depth_v2_labeled.mat'
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
    images = np.transpose(mat["images"], (0, 3, 2, 1))#(0, 2, 3, 1))
    depths = np.transpose(mat["depths"], (0, 2, 1))

    # Split into train/val (NYU depth commonly uses 795/654 split or similar)
    # For demonstration, let's do a simple split:
    num_samples = images.shape[0]
    train_count = int(0.8 * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    # Create dataset objects
    rgb_transform = T.Compose([
        T.Resize((240, 320)),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    depth_transform = T.Compose([
        T.Resize((240, 320)),
        #T.ToTensor(), 
        #T.Normalize(mean=[2.0],std=[1.0])
    ])

    train_dataset = NYUMatDataset(images[train_indices], depths[train_indices],
                                  transform=rgb_transform, depth_transform=depth_transform)
    val_dataset = NYUMatDataset(images[val_indices], depths[val_indices],
                                transform=rgb_transform, depth_transform=depth_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    #model = SimpleDepthEstimationNet()
    model = UNetDepthEstimator()
    train_depth_estimator(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda')
