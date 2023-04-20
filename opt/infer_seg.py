import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms as T
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torch import nn
from torchvision.models import resnet50

from os import listdir
from os.path import isfile, join

import datetime

sys.path.append("..")
from segment_anything import SamPredictor, sam_model_registry

# COCO classes
CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def get_transforms(img_size):
    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [
            # T.ToTensor(),
            # T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def load_images(file_name):
    im = Image.open(file_name)
    # im = np.array(im)
    # print(im)
    return im

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

def get_models(detr_version, model_type, checkpoint):
    # TODO Config and get model func
    # model = torch.hub.load("facebookresearch/detr", detr_version, pretrained=True)
    # model.eval()

    model = DETRdemo(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
    model.load_state_dict(state_dict)
    model.eval()

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    return model, predictor

masks_prev = None
boxes_prev = None
out_img_prev = None

def inference(im, transform, model_obj, model_seg, class_id , folder="ckpt_arf/llff/room_19/masked/", filename='DJI_20200226_143948_113.JPG'):
    img = transform(im).unsqueeze(0)
    # print("inside SAM img shape = ", img.shape)

    # propagate through the model
    outputs = model_obj(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.6

    if(probas[keep].shape[0]==0):
        return masks_prev, boxes_prev, out_img_prev

    #print(probas[keep].argmax(dim = 1))

    tv_idx = class_id
    out_class = probas[keep].argmax(dim = 1)
    idx = np.isin(out_class, class_id)
    #print(idx, out_class)
    #idx = == tv_idx
    probs = probas[keep][idx]
    #print("probs.shape = ", probs.shape, keep)
    #print("BOX SHAPE: " , outputs["pred_boxes"][0, keep][idx].shape)
    #box_idx = probs[keep][]
    # print("probs[box_idx_tv] = ", probs[box_idx_tv][72])
    
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep][idx], im.size)

    im = np.array(im)

    model_seg.set_image(im)
    
    transformed_boxes = model_seg.transform.apply_boxes_torch(
        bboxes_scaled.detach(), im.shape[:2]
    )
    #print("GGG  : ", transformed_boxes.shape, im.shape)
    masks, _, _ = model_seg.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.cuda(),
        multimask_output=False,
    )

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis('off')
    # plt.show()
    name = folder + 'orig_' + filename[-7:-4] + '.png'
    plt.savefig(name)

    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        
        now = datetime.datetime.now()
    name = folder + 'masked_' + str(now) + '.png'
    plt.savefig(name)
    plt.close()
        
    for box in bboxes_scaled:
        show_box(box.detach().cpu().numpy(), plt.gca())

    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    masks_prev, boxes_prev, out_img_prev = masks, bboxes_scaled, mat

    # plt.axis('off')
    # plt.show()
    # name = folder + 'test_' + filename[-7:-4] + '.png'
    # plt.savefig(name)
    # plt.close()
    return masks, bboxes_scaled, mat

def sam_wrapper(img, class_idx):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    detr_version = "detr_resnet50"
    img_size = 800

    transforms = get_transforms(img_size)
    obj_model, seg_model = get_models(detr_version, model_type, sam_checkpoint)

    masks, boxes, out_img = inference(img, transforms, obj_model, seg_model, class_idx)

    return masks, boxes, out_img

def main():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    detr_version = "detr_resnet50"
    img_size = 800

    folder = "/home/moneish/dish/vlr_project/ARF-svox2/data/llff/room/images/"
    out_folder = "/home/moneish/dish/vlr_project/ARF-svox2/out_boxes/"

    # files = [f for f in listdir(folder) if isfile(join(folder, f))]

    # files = ['DJI_20200226_143948_113.JPG', 'DJI_20200226_143902_603.JPG', 'DJI_20200226_143939_277.JPG', 'DJI_20200226_143938_168.JPG', 'DJI_20200226_143905_272.JPG', 'DJI_20200226_143906_682.JPG']

    files = ['DJI_20200226_143948_113.JPG']

    for file_name in files:
        file = join(folder, file_name)
        img = load_images(file)
        # print("Original img size = ", img.size)
        transforms = get_transforms(img_size)
        obj_model, seg_model = get_models(detr_version, model_type, sam_checkpoint)

        masks, boxes, out_img = inference(img, transforms, obj_model, seg_model, out_folder, file_name)

        masks = masks.squeeze()
        # print(masks.shape)
        a = torch.nonzero(masks)
        # print(a.shape)
        img = torch.tensor(np.array(img))
        # print(img.shape)
        img = img[a[:, 0], a[:, 1], :]
        # print(img.shape)


if __name__ == "__main__":
    main()