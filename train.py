import argparse
import os
import wandb

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from tqdm import tqdm, trange

from dataset import CustomCocoDataset, ResizeWithBoxes, ToTensorWithBoxes, CustomCompose
from utils import save_checkpoint


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    args = parser.parse_args()

    # Logger
    if args.use_wandb:
        wandb.init(
            project="Custom COCO Training",
            dir='./wandb_log',
        )

    # Data
    transform = CustomCompose([
        ResizeWithBoxes((512, 384)),
        ToTensorWithBoxes(),
    ])
    tr_ds = CustomCocoDataset(image_path='./data/train/image/', annot_path='./data/train.json', transform=transform)
    te_ds = CustomCocoDataset(image_path='./data/test/image/', annot_path='./data/test.json', transform=transform)
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
    model = fasterrcnn_resnet50_fpn_v2(
        # weights_backbone=ResNet50_Weights,
        # trainable_backbone_layers=3,
        num_classes=2,
    )
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # Train and validation
    checkpoint_dir = 'checkpoint/'
    best_valid_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.
        valid_loss = 0.

        # train
        for images, targets in tqdm(tr_dl):
            images = list(img.to(device) for img in images)

            boxes = targets
            targets = list()
            for box in boxes:
                target = {
                    'boxes': box.to(device),
                    'labels': torch.ones((box.size(0),), dtype=torch.int64, device=device),
                }
                targets.append(target)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())  # weights are all 1.0

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss = train_loss / len(tr_dl)
        print(train_loss)

        # eval
        with torch.no_grad():
            for images, boxes in tqdm(te_dl):
                images = list(img.to(device) for img in images)

                targets = list()
                for box in boxes:
                    target = {
                        'boxes': box.to(device),
                        'labels': torch.ones((box.size(0),), dtype=torch.int64, device=device)
                    }
                    targets.append(target)

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                valid_loss += losses.item()

            valid_loss = valid_loss / len(te_dl)
            print(valid_loss)

        if args.use_wandb:
            wandb.log({"train-loss": train_loss, "valid-loss": valid_loss, "epoch": epoch + 1})

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
