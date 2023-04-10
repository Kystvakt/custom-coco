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
        self.annotations = annot_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

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
        self.size = size

    def __call__(self, image, boxes):
        w, h = image.size
        new_w, new_h = self.size

        w_scale = new_w / w
        h_scale = new_h / h

        resized_image = Resize(self.size)(image)

        boxes_resized = boxes.clone()
        boxes_resized[:, 0] *= w_scale  # x1
        boxes_resized[:, 1] *= h_scale  # y1
        boxes_resized[:, 2] *= w_scale  # x2
        boxes_resized[:, 3] *= h_scale  # y2

        return resized_image, boxes_resized


class ToTensorWithBoxes:
    def __call__(self, image, boxes):
        image_tensor = ToTensor()(image)
        return image_tensor, boxes


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class MyDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        # self.annotation = annotation
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

        # labels (only target: hair or background)
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

            # resize bounding boxes
            # ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
            # ratio_width, ratio_height = ratios
            ratio_width, ratio_height = 0.25, 0.250326
            # img.size should be (2048, 1534)
            # rescaled_img.size should be (512, 384)

            # annots = coco_annotation.copy()
            # for annot in annots:
            #     boxes = annot['bbox']
            #     ratio = [ratio_width, ratio_height, ratio_width, ratio_height]
            #     if len(boxes) != 0:
            #         scaled_boxes = list(boxes[i] * ratio[i] for i in range(4))
            #         scaled_boxes = torch.Tensor(scaled_boxes)
            #     else:
            #         scaled_boxes = torch.zeros((0, 4), dtype=torch.float32)
            #     # scaled_boxes = boxes * [ratio_width, ratio_height, ratio_width, ratio_height]
            #     annot['bbox'] = scaled_boxes
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
