import torch
from torch import nn
from torchsummary import summary
from thop import profile
from thop import clever_format

from torchvision.models import resnet18, resnet50
from model.CMRNet import CMRNet
from torchvision.models import resnet18, resnet50

model = CMRNet()
# model = resnet18()
# model = resnet50()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# features = nn.Sequential(*list(model.children())[:10])
# summary(features, (3, 256, 256))

summary(model, (3, 256, 256))
model_name = 'cls'
input = torch.randn(1, 3, 256, 256)

input = input.to(device)

flops, params = profile(model, inputs=(input, ), verbose=True)

print("model: %s | params: %.2f (M)| FLOPs: %.2f (G)"%(model_name, params/(1000**2), flops/(1000**3)))

# 使用 thop.clever_format 格式化结果，提高结果的可读性
macs, params = clever_format([flops, params], "%.3f")