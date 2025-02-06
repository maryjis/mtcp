from queue import Queue
import numpy as np

class EarlyStopper:
    def __init__(
        self, 
        patience=1, 
        eps=0, 
        smoothing="no_smoothing", #"simple", "running_average" 
        smoothing_period=10,
        direction="min", #"min", "max"
    ):
        self.patience = patience
        self.eps = eps
        self.counter = 0
        assert direction in ["min", "max"], "Choose direction from 'min', 'max'"
        self.direction = direction
        self.optimal_value = float('inf') if direction == "min" else -float('inf')
        assert smoothing in ["no_smoothing", "running_average"], "Choose smoothing from 'no smoothing', 'running_average'"
        self.smoothing = smoothing
        self.smoothing_period = smoothing_period

        if smoothing == "running_average":
            self.last_values = Queue(smoothing_period)

    def early_stop(self, value):
        if self.smoothing == "running_average":
            if self.last_values.full():
                self.last_values.get()
            self.last_values.put(value)

            if self.last_values.qsize() < self.smoothing_period:
                return False
            
            value = np.mean(self.last_values.queue)

        if (self.direction == "min" and value < self.optimal_value) or \
            (self.direction == "max" and value > self.optimal_value):
            self.optimal_value = value
            self.counter = 0
            
        elif (self.direction == "min" and value > (self.optimal_value + self.eps)) or \
            (self.direction == "max" and value < (self.optimal_value - self.eps)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
                
        return False