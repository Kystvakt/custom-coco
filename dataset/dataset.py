import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from pycocotools.coco import COCO
from PIL import Image


class CustomCocoDataset(Dataset):
    def __init__(self, image_path, annot_path, transform=None):
        self.root = image_path
        self.coco = COCO(annot_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(coco_annotation)
        targets = list()
        for i in range(num_objs):
            target = coco_annotation[i]['bbox']
            targets.append(target)
        targets = torch.tensor(targets)

        if self.transform is not None:
            img, targets = self.transform(img, targets)

        return img, targets


class ResizeWithBoxes:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, image, boxes=None):
        w, h = image.size
        new_h, new_w = self.size  # do not be confused!

        w_scale = new_w / w
        h_scale = new_h / h

        resized_image = Resize(self.size)(image)

        if boxes is not None:
            boxes_resized = boxes.clone()
            boxes_resized[:, 0] *= w_scale  # x1
            boxes_resized[:, 1] *= h_scale  # y1
            boxes_resized[:, 2] *= w_scale  # x2
            boxes_resized[:, 3] *= h_scale  # y2

            for i in range(len(boxes_resized)):
                if boxes_resized[i, 0] == boxes_resized[i, 2] or boxes_resized[i, 1] == boxes_resized[i, 3]:
                    print(image)
                    print(boxes[i])
        else:
            boxes_resized = None

        return resized_image, boxes_resized


class ToTensorWithBoxes:
    def __call__(self, image, boxes=None):
        image_tensor = ToTensor()(image)
        return image_tensor, boxes


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class MyDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # bounding boxes for objects
        boxes = list()
        for i in range(num_objs):
            boxes.append(coco_annotation[i]['bbox'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # labels (only one target: hair or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # image id to tensor
        img_id = torch.tensor([img_id])

        # annotation
        annots = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
        }

        if self.transforms is not None:
            rescaled_img = self.transforms(img)

            ratio_width, ratio_height = tuple(float(s)/float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
            scaled_boxes = list()
            for box in annots['boxes']:
                ratio = [ratio_width, ratio_height, ratio_width, ratio_height]
                if len(boxes) != 0:
                    scaled_box = torch.Tensor(list(box[i] * ratio[i] for i in range(4)))
                scaled_boxes.append(scaled_box)
            scaled_boxes = torch.vstack(scaled_boxes)
            annots['boxes'] = scaled_boxes

            img = rescaled_img
        else:
            for annot in annots:
                boxes = annot['boxes']
                if len(boxes) != 0:
                    boxes = torch.Tensor(boxes)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                annot['boxes'] = boxes

        return img, annots

    def __len__(self):
        return len(self.ids)
