import torch
import torchvision
import numpy as np
import os
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from my_utils import get_transform, my_dataloader, get_pretrain_model
from engine import train_one_epoch, evaluate
import utils

is_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if is_cuda else 'cpu')
device = torch.device('cpu')

path = "../cv2_data/PennFudanPed"

# create dataloader
dataset = my_dataloader(path, get_transform(True))
dataset_test = my_dataloader(path, get_transform(False))

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
#                                           collate_fn=utils.collate_fn)
# data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=True, num_workers=4,
#                                                collate_fn=utils.collate_fn)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                          collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=True,
                                               collate_fn=utils.collate_fn)

# build model
num_classes = 2
model = get_pretrain_model('1', num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# save_model
state = model.state_dict() 
for key in state:
    state[key] = state[key].clone().cpu()

torch.save(state, './try_1.pt')

print("That's it!")

#"https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"


