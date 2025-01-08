import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn

model = nn.Conv2d(3, 3, kernel_size=3).cuda()
x = torch.randn(1, 3, 224, 224).cuda()
output = model(x)
print(output.shape)
