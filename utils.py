import torch


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
