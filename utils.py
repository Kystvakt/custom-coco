import sys
import torch
from time import time


def save_checkpoint(epoch, model, train_loss, val_loss, optimizer=None, lr_scheduler=None, filepath='checkpoint/'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    torch.save(checkpoint, filepath)


def progressbar(current, total, message=None):
    total_length = 65

    current_length = int(total_length * (current + 1) / total)
    rest_length = int(total_length - current_length) - 1

    sys.stdout.write(" [")
    sys.stdout.write("=" * current_length)
    sys.stdout.write(">")
    sys.stdout.write(" " * rest_length)
    sys.stdout.write("] ")

    sys.stdout.write(f"{current+1:>{len(str(total))}}/{str(total)}")

    # sys.stdout.write(f"Step: {step_time:.2f} | Total: {total_time:.2f}")
    sys.stdout.write(f"")
    if message:
        sys.stdout.write(" | " + message)

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
