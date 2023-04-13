import argparse
import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from dataset import CustomCocoDataset, ResizeWithBoxes, ToTensorWithBoxes, CustomCompose
from utils import save_checkpoint, progressbar


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--continue-from", default=None, type=str, help="checkpoint file path")
    args = parser.parse_args()

    # Logger
    if args.use_wandb:
        wandb.init(
            project="Custom COCO Training",
            dir='./wandb_log',
        )

    # Data
    transform = CustomCompose([
        ResizeWithBoxes((384, 512)),  # takes (height, width)
        ToTensorWithBoxes(),
    ])
    tr_ds = CustomCocoDataset(image_path='./data/train/image/', annot_path='./data/train.json', transform=transform)
    te_ds = CustomCocoDataset(image_path='./data/test/image/', annot_path='./data/test.json', transform=transform)
    if args.sample:
        tr_ds = Subset(tr_ds, np.arange(1))
        te_ds = Subset(te_ds, np.arange(1))
        args.batch_size = 1
        args.num_epochs = 200

    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count()//2,
        collate_fn=collate_fn
    )
    te_dl = DataLoader(
        te_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count()//2,
        collate_fn=collate_fn
    )
    print("Train dataset size:", len(tr_ds))
    print("Validation dataset size:", len(te_ds))

    # CUDA
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    model = fasterrcnn_resnet50_fpn_v2(num_classes=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    if args.continue_from is not None:
        checkpoint = torch.load(args.continue_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']

    # Train and validation
    checkpoint_dir = 'checkpoint/best/'
    best_valid_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.
        valid_loss = 0.

        # train
        # TODO: implement learning rate scheduler
        for batch_idx, (images, boxes) in enumerate(tr_dl):
            images = list(img.to(device) for img in images)

            targets = list()
            for box in boxes:
                target = {
                    'boxes': box.to(device),
                    'labels': torch.ones((box.size(0),), dtype=torch.int64, device=device),
                }
                targets.append(target)

            # training losses
            loss_dict = model(images, targets)
            train_loss_classifier = loss_dict['loss_classifier']
            train_loss_box_reg = loss_dict['loss_box_reg']
            train_loss_objectness = loss_dict['loss_objectness']
            train_loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # progress bar
            train_loss += losses.item()
            log = f"Train Loss: {losses.item():.3f}"
            progressbar(batch_idx, len(tr_dl), log)

        train_loss = train_loss / len(tr_dl)

        # eval
        with torch.no_grad():
            for batch_idx, (images, boxes) in enumerate(te_dl):
                images = list(img.to(device) for img in images)

                targets = list()
                for box in boxes:
                    target = {
                        'boxes': box.to(device),
                        'labels': torch.ones((box.size(0),), dtype=torch.int64, device=device)
                    }
                    targets.append(target)

                # validation losses
                loss_dict = model(images, targets)
                valid_loss_classifier = loss_dict['loss_classifier']
                valid_loss_box_reg = loss_dict['loss_box_reg']
                valid_loss_objectness = loss_dict['loss_objectness']
                valid_loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']

                losses = sum(loss for loss in loss_dict.values())

                # progress bar
                valid_loss += losses.item()
                log = f"Valid Loss: {losses.item():.3f}"
                progressbar(batch_idx, len(te_dl), log)

            valid_loss = valid_loss / len(te_dl)

        # logging
        if args.use_wandb:
            wandb.log({
                "train_loss_classifier": train_loss_classifier,
                "train_loss_box_reg": train_loss_box_reg,
                "train_loss_objectness": train_loss_objectness,
                "train_loss_rpn_box_reg": train_loss_rpn_box_reg,
                "valid_loss_classifier": valid_loss_classifier,
                "valid_loss_box_reg": valid_loss_box_reg,
                "valid_loss_objectness": valid_loss_objectness,
                "valid_loss_rpn_box_reg": valid_loss_rpn_box_reg
            })

        # save checkpoint
        ckptpath = os.path.join('checkpoint/', 'checkpoint.pth')
        save_checkpoint(epoch + 1, model, train_loss, valid_loss, optimizer, filepath=ckptpath)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint.pth")
            save_checkpoint(epoch + 1, model, train_loss, valid_loss, optimizer, filepath=checkpoint_path)
            print(f"New best loss: {valid_loss}")
            print("Checkpoint saved.")

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
