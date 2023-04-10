import glob

import torch.cuda
from PIL import Image
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import Compose, ToTensor, Resize


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load model
model = fasterrcnn_resnet50_fpn_v2(pretrained=False, num_classes=2)
model.to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('checkpoint/checkpoint_epoch_12.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

model.eval()

filelist = glob.glob('constant1_*.jpg', root_dir='../data/train/image/')

for img in filelist:
    x = Image.open('./data/train/image/' + img)
    transform = Compose([
        ToTensor(),
        Resize((512, 384), antialias=True),
    ])
    x = transform(x)
    x = x.unsqueeze(0).to(device)
    y_hat = model(x)

    print(len(y_hat))
    break
