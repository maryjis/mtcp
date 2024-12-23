class EarlyStopper:
    def __init__(self, patience=1, eps=0):
        self.patience = patience
        self.eps = eps
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        elif validation_loss > (self.min_validation_loss + self.eps):
            self.counter += 1
            if self.counter >= self.patience:
                return True
                
        return False