import torch


class EarlyStopper:
    """Stop training after validation loss stops improving."""
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        assert min_delta >= -1e-4, "A large negative min_delta will never trigger early stopping"
        self.min_delta = min_delta
        self.min_loss = 100
        self.wait = 0
        self.stop_training = False
        self.verbose = verbose

    def __call__(self, loss):
        """Return whether the current validation loss improved."""
        if loss <= (self.min_loss - self.min_delta):
            if loss < self.min_loss:
                self.min_loss = loss
            self.wait = 0
            return True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"----Early stopping with loss {self.min_loss:0.6f}----")
            return False


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({'epoch': epoch,  'loss': loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
