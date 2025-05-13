class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode="max"):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change to qualify as improvement.
        """

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        if mode not in {'min', 'max'}:
            raise ValueError("mode should be 'min' or 'max'")
    
    def __call__(self, current_score, model):
        # Adjust score depending on mode
        score = current_score if self.mode == "max" else -current_score

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
    
    def load_best_model(self, model):
        """Restore model to the best saved state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)