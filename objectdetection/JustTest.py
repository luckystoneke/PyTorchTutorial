import torch
import torchvision

from PIL import Image

# img = Image.open('PennFudanPed/PNGImages/FudanPed00001.png')
#
# mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
#
# mask.putpalette([
#     0, 0, 0,  # black background
#     255, 0, 0,  # index 1 is red
#     255, 255, 0,  # index 2 is yellow
#     255, 153, 0,  # index 3 is orange
# ])
#
#
# Image._show(img)
# Image._show(mask)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
import transforms as T
from ObjDetectionFinetuning import PennFudanDataset, get_transform


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
model = get_model_instance_segmentation(2)
# model = torch.load(r'mask_R_CNN.pth')
model.load_state_dict(torch.load(r'mask_R_CNN.pth'))
# pick one image from the test set
# img, _ = dataset_test[0]
img_raw = Image.open('./TestImages/FudanPed00005.png').convert("RGB")
img = torchvision.transforms.functional.to_tensor(img_raw)
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
print(prediction)
im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
Image._show(im)
im2 = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
Image._show(im2)
print("That's it!")
