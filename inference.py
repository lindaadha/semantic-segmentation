from torch import nn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet import ResUnet
from torchvision import transforms
import numpy as np
from PIL import Image
import glob, base64
from io import BytesIO

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
    # return model

def convert_base64 (img):
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)

    return im_b64

checkpoint_path = 'model_weight/chkpoint_'
model = ResUnet(3)
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
model, optimizer, start_epoch, valid_loss_min = load_ckp(checkpoint_path, model, optimizer)


get_train_transform = A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensorV2()
        ])


img_name = 'test/test.png'

pillow_image = Image.open(img_name)
image = np.array(pillow_image)
image = image[:, :, :3]
print(image.shape)
image = get_train_transform(image=image)['image'].unsqueeze(0)
print(image.shape)

image = torch.autograd.Variable(image, volatile=True)
print(image.shape)

output = model(image)
print(output.shape)

image_segmentation=output[0].data.cpu()
image_segmentation= transforms.ToPILImage()(image_segmentation)
print(image_segmentation.size)
# image_segmentation.show()

base64_img = convert_base64(image_segmentation)
print(base64_img)

