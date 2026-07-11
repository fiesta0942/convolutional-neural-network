# Implementation of MobileNet V2
# Objective: Understand 'depthswise separable convolution' with adding 'expansion and projection'
# Minsoo Kang (Kyung Hee University, Department of Mathematics)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914,0.4822,0.4465),std=(0.2470,0.2435,0.2616))])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False,download=True,transform = transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0 )


#######################################################################################################
# 2. Basic Convolution Block: Input -> Conv2D -> BatchNorm -> ReLU
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2D, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
    
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu6(out)
        
        
        return out
        
    
    
# 3. Depthwise Separable Convolution
# Input -> Depthwise 3x3 conv -> B.N and ReLU -> Pointwise 1x1 conv -> B.N and ReLU
# Not used in MobileNet V2, but for the sake of study


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, final_out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparable, self).__init__()
        depthwise_out_channels = in_channels
        
        self.depthwiseconv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = depthwise_out_channels,     # Depthwise conv does NOT change the # of channels
            kernel_size = 3,                # 3x3 conv
            stride = 1,
            padding = 1,                    # Original CIFAR-10 is 32 x 32. To keep spatial dimension same, s = p = 1
            groups = in_channels            # This make the convolution separate for each channel (Definition of depthwise conv)
        )
        
        self.bn1 = nn.BatchNorm2d(depthwise_out_channels)
        
        
        self.pointwiseconv = nn.Conv2d(
            in_channels = depthwise_out_channels,
            out_channels = final_out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.bn2 = nn.BatchNorm2d(final_out_channels)
        
    def forward(self, x):
        out = self.depthwiseconv(x)
        out = self.bn1(out)
        out = F.relu6(out)
        
        out = self.pointwiseconv(out)
        out = self.bn2(out)
        out = F.relu6(out)
        
        return out
    
    
# 4. Bottleneck Block
# Input -> 1x1 Expansion -> 3x3 Depthwise -> 1x1 Porjection -> Optional Shortcut -> Output

class bottleneck(nn.Module):
    def __init__ (self , in_channels, final_out_channels, stride, expansion_factor):
        super(bottleneck, self).__init__()
        
        self.shortcut = (stride == 1 and in_channels == final_out_channels)     # bottleneck에서 입력과 출력 dimension이 같다면 shortcut 사용
        
        expansion_out_channels = in_channels * expansion_factor     # Important: Origianl channel is "expaned" at 1x1 expansion
        
        self.expansion = nn.Conv2d(
            in_channels = in_channels,
            out_channels = expansion_out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0     # Expansion only changes channel, NOT spatial size
        )
        
        self.bn1 = nn.BatchNorm2d(expansion_out_channels)
        
        self.depthwiseconv = nn.Conv2d(
            in_channels = expansion_out_channels,
            out_channels = expansion_out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            groups = expansion_out_channels
        )
        
        self.bn2 = nn.BatchNorm2d(expansion_out_channels)
        
        self.pointwiseconv = nn.Conv2d(
            in_channels = expansion_out_channels,
            out_channels = final_out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.bn3 = nn.BatchNorm2d(final_out_channels)
        
    def forward(self,x):
        out = self.expansion(x)
        out = self.bn1(out)
        out = F.relu6(out)
        
        out = self.depthwiseconv(out)
        out = self.bn2(out)
        out = F.relu6(out)
        
        out = self.pointwiseconv(out)
        out = self.bn3(out)
        
        if self.shortcut:
            out = x + out
        
        return out 
    
# 5. Stack Bottleneck block following "t,c,n,s" from the article
# Input -> Depthwise Convolution -> Several Bottleneck Layer -> Pointwise Convolution -> Avg Pooling -> Softmax
# To stack up the bottleneck layer, use self.cfgs = [t,c,n,s]
# t: expansion factor, c: output channel, n: number of stacked bottleneck, s: stride of first block
    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes = 10):
        super(MobileNetV2, self).__init__()
        
        # Designate the expansion factor, output channel, epoch number and stirde for "each bottleneck layer"
        self.cfgs = [
            [1,16,1,1],
            [6,24,2,1], # Original article used [6,24,2,2]. However to match the dimension of CIFAR-10, I've used stride 1 
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1],
        ]               # Totally, we stacked up 17 layers of bottleneck
        
        self.first_conv = BasicConv2D(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        
        self.bottlenecks = self.make_layers(in_channels = 32)
        
        self.last_conv = BasicConv2D(
            in_channels = 320,
            out_channels = 1280,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Linear(1280, num_classes)
        
        
    def make_layers(self, in_channels):
        layers = []
        
        for t,c,n,s in self.cfgs:
            out_channels = c
            
            for i in range(n):
                stride = s if i == 0 else 1
                
                layers.append(
                    bottleneck(
                        in_channels = in_channels,
                        final_out_channels = out_channels,
                        stride = stride,
                        expansion_factor = t
                    )
                )
                in_channels = out_channels
                
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.first_conv(x)
        out = self.bottlenecks(out)
        out = self.last_conv(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out
    
    
# 5. Network 통과 이후 후처리

model = MobileNetV2(num_classes = 10).to(device)

# 5-1 Softmax Activation
criterion = nn.CrossEntropyLoss()       # Evaluate the difference between output and labels. PyTorch does the softmax and negative log likelihood

# 5-2 Update the parameters using Gradient Descent, Mini-batch G.D, G.D with Momentum, Adam Optimization
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)    # learning rate = 0.001

# 5-3 With the softmax and Adam i've made, do the "actual" work for all mini - batch "five time" (5 epoch)

num_epochs = 5 
for epoch in range(num_epochs):
    # Training
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        
        images = images.to(device)  # 현재 사용하는 Device: CPU -> model, image, label 모두 cpu 로 보내야함
        labels = labels.to(device)
        
        outputs = model(images)     # model = MobileNetV2 으로 설정함. 즉 여기서 forward propagation 진행
        loss = criterion(outputs, labels)  
        
        optimizer.zero_grad()       # Mini-batch G.D에선 매 batch 마다 새로운 gradient를 구함. PyTorch에선 기본적으로 계산된 gradient가 누적되므로, zero_grad를 통해 gradient를 리셋
        loss.backward()             # Backward Propagation 진행
        optimizer.step()            # Update Parameters 진행 using Adam Optimization
            
        total_loss += loss.item() * images.size(0)      # 모든 mini batch에 대해 평균 loss
        _, predicted = outputs.max(1)                   # Accuracy Calculation
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss = total_loss / total
    train_acc = correct / total * 100 
        
    
    # Test Set
    model.eval()
    
    test_total_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs,labels)
            
            test_total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
        test_loss = test_total_loss / test_total
        test_acc = test_correct / test_total * 100
    

    print(f"Epoch [{epoch+1} / {num_epochs}]" f"Train Loss: {train_loss:.4f}" f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}" f"Test Accuracy: {test_acc:.2f}%")