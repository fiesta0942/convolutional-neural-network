# Simple ResNet using Pytorch
# Using fundamental CNN layers(Padding, Strided, Pooling), practice implementing ResNet
# Minsoo Kang (Kyung Hee University, Department of Mathematics)


# 1. Load CIFAR-10 Dataset using PyTorch.
import torch
import torchvision                
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # 노트북이라 CPU 사용

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914,0.4822,0.4465),std=(0.2470,0.2435,0.2616))])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False,download=True,transform = transform)

# ToTensor: Originally the image is 32 x 32 x 3(height, width, channel). However, Pytorch need 3 x 32 x 32(channel x heigth x width)
# ToTensor changes the dimension order to channel x height x width
# ToTensor also makes the value of pixel from "0 ~ 255" to "0.0 ~ 1.0"
# Normalize makes the mean value of pixel around 0 and fix variance

# .Compose makes both action work (First ToTensor, then Normalize)

# Q. What is the size of one image?
image, label = train_dataset[0]
print("Image Shape:", image.shape)
print("Label:", label)
print("class name:", train_dataset.classes[label])

# Use DataLoader to make mini-batch of images for mini batch Gradient Descent
batch_size = 128        # one batch has 128 images
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 0)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle = False, num_workers = 0)

####################################################################################################

# 2. Make Residual Blcok
# Residual block: Input x -> Conv 3x3, ReLU -> Conv 3x3 -> Add shortcut -> ReLU -> Output

class ResidualBlock(nn.Module):         #nn.Module: Pytorch 내부 자체 model. Convolution Layer들의 parameter를 알아서 관리
    def __init__(self, in_channels, out_channels, stride = 1):  # 실제 사용할때, block = ResidualBlock(64,128,stride = 2) 같이 사용. 즉, 실사용때 channel 수를 내가 고를 수 있게 함
        
        super(ResidualBlock, self).__init__()
        
        # Plain Path
        self.conv1 = nn.Conv2d(             # First Convolution Layer
            in_channels = in_channels,      # number of Input Channels
            out_channels = out_channels,    # number of Output Channels
            kernel_size = 3,                # Uses 3 x 3 convolution filter
            stride = stride,
            padding = 1                     # Same Padding -> match the dimensions of input and output
        )
        
        self.conv2 = nn.Conv2d(             # Second Convolution Layer
            in_channels = out_channels,     # Input, Output Channel가 같도록 유지
            out_channels = out_channels,
            kernel_size = 3,                
            stride = 1,
            padding = 1
        )
        
        # Shortcut Path
        self.shortcut = nn.Sequential()     # Identity Function (Input X를 ReLU 직전에 "그대로" 가져온다)
        
        # Use 1x1 convolution to match dimensions for output = output + identity
        if stride != 1 or in_channels != out_channels:      # 1. stride가 1이 아니라면, plain path에서 height,width가 줄어든다 -> shortcut에서도 똑같이 줄여야함
            self.shortcut = nn.Sequential(                  # 2. main path에서 input - output channel이 달라졌다 -> shortcut에서도 channel을 맞춰야함
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,  # 1x1 convolution
                    stride = stride,
                    padding = 0
                )
            )    
            
    def forward(self, x):               # 실제로 shortcut을 사용해 Residual Block 계산을 진행
        identity = self.shortcut(x)     # Input X와 output (after 2 convolution)의 size가 같으면 x. 아니라면 1x1 conv(x) 사용
        
        out = self.conv1(x)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = out + identity
        out = F.relu(out)
        
        return out
    
    
####################################################################################################

# 2-1. 실제 작동 확인

####################################################################################################

# 3. Make ResNet by stacking up Residual Block
# For CIFAR-10, Input -> conv1 -> layer1 -> layer2 -> layer3 -> layer4 -> avg pooling -> fully connected -> output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):    # block: Which residual block to use? , # num_blocks = How may residual block in each layer? [2,2,2,2]라면 layer 1,2..에 2개를 쌓는다
        super(ResNet, self).__init__()                          
        
        self.in_channels = 64
        
        # Conv 1
        # Note CIFAR-10 Input = [batch_size, 3, 32, 32] 
        self.conv1 = nn.Conv2d(
            in_channels = 3,       # Input image has 3 RGB Channel
            out_channels = 64,     # After convolution, it becomes 64 channel, 64 x 32 x 32
            kernel_size = 3,
            stride = 1,
            padding = 1            # For s = 1, p = 1 -> spatial size is preserved
        )
        
        # Res Layer 1,2,3,4. Increase Channel / Decrease Height, Width
        self.layer1 = self.make_layer(block, out_channels = 64, num_blocks = num_blocks[0], stride = 1) # 64 channel 유지. 64 x 32 x 32
        self.layer2 = self.make_layer(block, out_channels = 128, num_blocks = num_blocks[1], stride = 2) # 128 channel로 증가 및 spatial size 감소. 128 x 16 x 16
        self.layer3 = self.make_layer(block, out_channels = 256, num_blocks = num_blocks[2], stride = 2) # 256 x 8 x 8
        self.layer4 = self.make_layer(block, out_channels = 512, num_blocks = num_blocks[3], stride = 2) # 512 x 4 x 4
        
        # Average Pooling Layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) # 512 x 1 x 1. 512개의 channel 각각에서 4x4 값의 평균
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes) # feature는 512개 but 구분 할 class is 10. 512 dimension feature vector -> 10 class
        
    def make_layer(self, block, out_channels, num_blocks, stride):  # Residual Block을 여러개 쌓아서 하나의 큰 layer를 만든다
                                                                    # self.make_layer(block, out_channels = 128, num_blocks = 2, stride = 2)라면 ResBlock(64,128,stride =2), ResBlock(128,128,stride = 1)을 묶음
        layers = []
        
        # Block을 쌓을 때, 첫번째 layer는 Conv1 때문에 사이즈가 변할 수 있음
        layers.append(
            block(in_channels = self.in_channels, out_channels = out_channels, stride = stride)
        )
        self.in_channels = out_channels #만약 첫번째 block에서 사이즈가 바뀌었다면, 이후 layer에서 바뀐 channel을 input으로 받아야함
        
        for _ in range(1, num_blocks):  #첫번째 block 이후에는 shape이 안바뀜.
            layers.append(
                block(in_channels = self.in_channels, out_channels= out_channels, stride = 1)
            )
        return nn.Sequential(*layers)   # *layers는 layer 안의 block들을 풀어줌. 즉 nn.Sequential(block1, block2 ,...) -> 1,2 순서대로 진행
    
    def forward(self, x):               # 실제 ResNet을 진행하는 함수
        out = self.conv1(x)
        out = F.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        
        out = torch.flatten(out, 1)
        out = self.fc(out)              
        
        return out
    
    
 ####################################################################################################
 # 4. Make ResNet-18 Model
 # Note that by using ResNet class, we can define how many blocks to stack up (e.g num_blocks = [2,2,2,2,] or num_blocks = [3,4,6,3])
 # We want to use particular number of stacked blocks which is [2,2,2,2] -> called ResNet-18 Model               
 
def ResNet18(num_classes = 10):
    return ResNet(block = ResidualBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    
# We can use this by model = ResNet18(num_classes = 10).to(device)


####################################################################################################
# 5. Network 통과 이후 후처리

model = ResNet18(num_classes = 10).to(device)

# 5-1 Softmax Activation
criterion = nn.CrossEntropyLoss()       # Evaluate the difference between output and labels. PyTorch does the softmax and negative log likelihood

# 5-2 Update the parameters using Gradient Descent, Mini-batch G.D, G.D with Momentum, Adam Optimization
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)    # learning rate = 0.001

# 5-3 With the softmax and Adam i've made, do the "actual" work for all mini - batch "five time" (5 epoch)

num_epochs = 5 
for epoch in range(num_epochs):
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)  # 현재 사용하는 Device: CPU -> model, image, label 모두 cpu 로 보내야함
        labels = labels.to(device)
        
        outputs = model(images)     # model = ResNet18으로 설정함. 즉 여기서 forward propagation 진행
        loss = criterion(outputs, labels)   # Forward prop 이후 나온 10개의 class score를 criterion을 이용해 loss (이미지 1개의)를 계산
        
        optimizer.zero_grad()       # Mini-batch G.D에선 매 batch 마다 새로운 gradient를 구함. PyTorch에선 기본적으로 계산된 gradient가 누적되므로, zero_grad를 통해 gradient를 리셋
        loss.backward()             # Backward Propagation 진행
        optimizer.step()            # Update Parameters 진행 using Adam Optimization
            
        total_loss += loss.item() * images.size(0)      # 모든 mini batch에 대해 평균 loss
        _, predicted = outputs.max(1)                   # Accuracy Calculation
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = total_loss / total
    train_acc = correct / total * 100
    print(f"Epoch [{epoch+1} / {num_epochs}]" f"Train Loss: {train_loss:.4f}" f"Train Accuracy: {train_acc:.2f}%")
            
   