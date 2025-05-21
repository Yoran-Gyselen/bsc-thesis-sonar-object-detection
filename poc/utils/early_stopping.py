import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode="max"):
        """
        Early stopping to stop training when the monitored score stops improving.

        Args:
            patience (int): Number of epochs to wait after last improvement.
            delta (float): Minimum change in score to qualify as an improvement.
            mode (str): One of "min" or "max". 
                        "min" means lower score is better (e.g., loss).
                        "max" means higher score is better (e.g., accuracy).
        """

        if mode not in {"min", "max"}:
            raise ValueError("mode should be either 'min' or 'max'")

        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score = None
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, current_score, model):
        """
        Call this method after each epoch to check if early stopping should occur.

        Args:
            current_score (float): The metric to monitor (e.g., loss or accuracy).
            model (nn.Module): The model to save if performance improves.
        """
        score = current_score if self.mode == "max" else -current_score

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
    
    def load_best_model(self, model):
        """
        Loads the best saved model weights into the given model and returns it.

        Args:
            model (nn.Module): The model to load the best state into.

        Returns:
            nn.Module: The model with best weights loaded (or as-is if no best saved).
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model