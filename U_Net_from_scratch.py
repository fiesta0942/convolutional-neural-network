# U-Net Implementation using Oxford - IIIT Pet segmentation
# Minsoo Kang (Kyung Hee University, Department of Mathematics)

import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = OxfordIIITPet(root="./data", split = "trainval", target_types = "segmentation", download = True)
test_data = OxfordIIITPet(root="./data", split = "test", target_types = "segmentation", download = True)


# 0. Check how does the image and mask look like?
image, mask = train_data[0]
mask_array = np.array(mask)

print(type(image))
print(type(mask))
print("Image Size:", image.size)
print("Mask size:", mask.size)

plt.figure(figsize = (10,4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Segmentation Mask")
plt.axis("off")
plt.show()

binary_mask = (mask_array == 1).astype(np.float32)

plt.imshow(binary_mask, cmap="gray")
plt.title("Binary Mask")
plt.axis("off")
plt.show()

# 1. Dataset 전처리
# 1-1. Resize original image and mask
image_size = 128            # Want to use 128 x 128 as input image and mask
image = TF.resize(image, [image_size, image_size])
mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)

# 1-2 Convert the original image to tensor
image_tensor = TF.to_tensor(image)

# 1-3 Convert the 3 class mask to binary mask
# Note the original segmentation mask has 3 pixel
# 1: pet, 2: background, 3: boundary
# We want to change it to pet: 1, background: 0, boundary: 0 or 1
mask_array = np.array(mask)

binary_mask = (mask_array == 1).astype(np.float32)

mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

# 2. 작성한 전처리를 "모든 Original Image, Mask"에 적용
class SegmentationDataset(Dataset):
    def __init__(self, root="./data", split="trainval", image_size = 128, download = True):
    
        self.dataset = OxfordIIITPet(root = root, split = split, target_types = "segmentation", download = download)
        
        self.image_size = image_size
        
    def __len__(self):
        return len(self.dataset)        # How man image in the dataset?
    
    def __getitem__(self, index):
        image, mask = self.dataset[index]
        
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        mask = np.array(mask)
        
        mask = (mask == 1).astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        
        return image, mask
    
# 3. 전처리 완료한 Data를 train / test로 나누기 using Dataset
train_data = SegmentationDataset(root="./data", split = "trainval", image_size = 128, download = True)
test_data = SegmentationDataset(root="./data", split = "test", image_size = 128, download = True)

batch_size = 8
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = False, num_workers=0)

# Original Data -> SegmentationDataset: 전처리 for 1 sample -> DataLoader: sends "batch_size" many samples to model

#####################################################################################################################

# Construct U-Net Architecture
# 1. Construct basic convolution layer that does 2 3x3 convolution and ReLU
# 2. Encoder: Stack 5 of them, each connected with MaxPooling to do the "dimesion deceasing" part of U-Net
# 3. Decoder: Stack 5 of them, now with each connected with Transpose Convolution to "dimension increasing" concatenating with encoder feature
# 4. Use 1x1 convolution to match the desired output dimension

# 1. Basic Convolution Layer (3x3 Conv -> B.N and ReLU -> 3x3 Conv -> B.N and ReLU)
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        return out
        
# ResNet, MobileNet에서 만든 basic convolution layer와 똑같이 생김

# 2. Encoder
# Stack 4 Basic Convolution with 1 bottleneck, connected using Max Pooling
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder1 = BasicConv(3,64)
        self.encoder2 = BasicConv(64,128)
        self.encoder3 = BasicConv(128,256)
        self.encoder4 = BasicConv(256,512)
        
        self.bottleneck = BasicConv(512,1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self,x):
        x1 = self.encoder1(x)
        p1 = self.maxpool(x1)
        
        x2 = self.encoder2(p1)
        p2 = self.maxpool(x2)
        
        x3 = self.encoder3(p2)
        p3 = self.maxpool(x3)
        
        x4 = self.encoder4(p3)
        p4 = self.maxpool(x4)
        
        final = self.bottleneck(p4)
        
        return x1,x2,x3,x4,final        #x1 ~ x4 is saved for skip connection
    
# 3. Decoder
# Input -> Transposed Convolution -> Concatenate with matching feature from Encoder(skip connection) -> Basic Convolution
# Stack 4 of it

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
          # 8x8 -> 16x16
        self.TransConv1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = BasicConv(
            in_channels=1024,
            out_channels=512
        )

        # 16x16 -> 32x32
        self.TransConv2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = BasicConv(
            in_channels=512,
            out_channels=256
        )

        # 32x32 -> 64x64
        self.TransConv3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = BasicConv(
            in_channels=256,
            out_channels=128
        )

        # 64x64 -> 128x128
        self.TransConv4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        
        self.decoder4 = BasicConv(
            in_channels=128,
            out_channels=64
        )
                                        
    def forward(self, bottleneck, x1, x2, x3, x4):
        out = self.TransConv1(bottleneck)
        out = torch.cat([out, x4], dim = 1)
        out = self.decoder1(out)
        
        out = self.TransConv2(out)
        out = torch.cat([out,x3], dim = 1)
        out = self.decoder2(out)
        
        out = self.TransConv3(out)
        out = torch.cat([out,x2], dim = 1)
        out = self.decoder3(out)
        
        out = self.TransConv4(out)
        out = torch.cat([out, x1], dim = 1)
        out = self.decoder4(out)
        
        return out
        
# 4. Add up the classes

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_conv = nn.Conv2d(
            in_channels = 64, out_channels = 1, kernel_size = 1
        )         
        
    def forward(self,x):
        x1,x2,x3,x4,bottleneck = self.encoder(x)
        
        out = self.decoder(bottleneck, x1,x2,x3,x4)
        
        out = self.final_conv(out)
        
        return out
    
#####################################################################################################################
# Training and Evaluate the model
model = UNet().to(device)

# Loss Function
criterion = nn.BCEWithLogitsLoss()  # Binary Classification. BCEwithLogits computes both sigmoid and binary cross entropy

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Dice Score
# In segmentation evaluation, Dice Score is used
# 예측 영역과 정답 영역이 얼마나 겹치는지 측정

def dice_score(outputs, masks, threshold = 0.5, epsilon = 1e-7):
    prob = torch.sigmoid(outputs)
    prediction = (prob > threshold).float()
    
    intersection = (prediction * masks).sum(dim=(1,2,3))
    total = (prediction.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)))
    dice = ( 2 * intersection + epsilon) / (total + epsilon)
    
    return dice.mean()

# Training / Evaluating
num_epoch = 10

for epoch in range(num_epoch):
    
    # Training (backward propagation)
    model.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        
        optimizer.zero_grad()       # Mini-batch G.D에선 매 batch 마다 새로운 gradient를 구함. PyTorch에선 기본적으로 계산된 gradient가 누적되므로, zero_grad를 통해 gradient를 리셋
        loss.backward()             # Backward Propagation 진행
        optimizer.step()            # Update Parameters 진행 using Adam Optimization
            
        train_loss += loss.item()
    average_train_loss = train_loss / len(train_loader)
    
    # Evaluating
    model.eval()
    
    test_loss = 0.0
    test_dice = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            
            test_loss += loss.item()
            test_dice += dice.item()
    average_test_loss = test_loss / len(test_loader)
    average_test_dice = test_dice / len(test_loader)
    
    print(
        f"Epoch [{epoch + 1}/{num_epoch}] | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Test Loss: {average_test_loss:.4f} | "
        f"Dice: {average_test_dice:.4f}"
    )