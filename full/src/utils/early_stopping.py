import torch

class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, path: str = 'checkpoint.pt', delta: float = 0):
        """Initialize early stopping object.
        
        Args:
            patience: Number of epochs to wait after validation loss stopped improving
            verbose: If True, prints message for each validation loss improvement
            path: Path for saving the checkpoint
            delta: Minimum change in monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Args:
            val_loss: Validation loss for current epoch
            model: Model to save checkpoint for
        """
        current_loss = val_loss

        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(model, val_loss)
        elif current_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Best loss: {self.best_loss:.6f}, Current loss: {current_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered after {self.patience} epochs without improvement')
        else:
            self.best_loss = current_loss
            self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, model: torch.nn.Module, val_loss: float):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss