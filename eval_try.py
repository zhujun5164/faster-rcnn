import torch
import torchvision
from my_utils import video_detection, img_dection

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

#build model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)

# video_detection(model, device)

img_path = './1.jpg'
img_dection(model, img_path, device)
print('a')