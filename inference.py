import glob

import numpy as np
import torch.cuda
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import Compose, ToTensor, Resize


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
checkpoint = torch.load('checkpoint/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
model.eval()

# Inference
filelist = glob.glob('*.jpg', root_dir='data/train/image/')

img_idx = 0
img_path = 'data/train/image/' + filelist[img_idx]

x = Image.open(img_path)
transform = Compose([
    Resize((512, 384), antialias=True),
    ToTensor(),
])
x = transform(x)
x = x.unsqueeze(0).to(device)
output = model(x)
y = np.array(x.cpu())
y = np.transpose(y[0], (1, 2, 0))
plt.imshow(y)
for (x1, y1, x2, y2) in output[0]['boxes']:
    x1 = x1.cpu().detach().numpy()
    y1 = y1.cpu().detach().numpy()
    x2 = x2.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()
    plt.plot([x1, x2], [y1, y2])
    # plt.scatter(x1, y1)
    # plt.scatter(x2, y2)
plt.show()
# for img in filelist:
#     x = Image.open('./data/train/image/' + img)
#     transform = Compose([
#         ToTensor(),
#         Resize((512, 384), antialias=True),
#     ])
#     x = transform(x)
#     x = x.unsqueeze(0).to(device)
#     y_hat = model(x)
#
#     print(y_hat.keys())
