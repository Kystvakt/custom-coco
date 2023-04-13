import glob

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from PIL import Image
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from dataset import CustomCompose, ResizeWithBoxes, ToTensorWithBoxes


# Set device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load model
model = fasterrcnn_resnet50_fpn_v2(num_classes=2)
model.to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load checkpoint
checkpoint = torch.load('checkpoint/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
model.eval()

# Inference
dir_path = 'data/train/image/'
filelist = glob.glob('*.jpg', root_dir=dir_path)

img_idx = 0
img_path = dir_path + filelist[img_idx]
x = Image.open(img_path)
transform = CustomCompose([
    ResizeWithBoxes((384, 512)),
    ToTensorWithBoxes(),
])

x = transform(x)[0]
x = x.unsqueeze(0).to(device)
output = model(x)

# for display purpose
y = np.array(x.cpu())
y = np.transpose(y[0], (1, 2, 0))

fig, ax = plt.subplots()
ax.imshow(y, origin='lower')
for (x1, y1, x2, y2) in output[0]['boxes']:
    x1 = x1.cpu().detach().numpy()
    y1 = y1.cpu().detach().numpy()
    x2 = x2.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, facecolor='none', edgecolor='r')
    ax.add_patch(rect)

plt.show()
