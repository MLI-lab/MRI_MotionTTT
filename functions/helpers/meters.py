import time
import torch
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count



class RunningAverageMeter(object):
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
class TimeMeter(object):
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

class TrackMeter_testing(object):
    def __init__(self,):
        self.reset()  

    def reset(self):
        self.val = []
        self.avg = 0
        self.std = 0

    def update(self, val,):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val.append(val)
        self.avg = np.mean(self.val)
        self.std = np.std(self.val)

class TrackMeter(object):
    def __init__(self, inc_or_dec='decaying'):
        self.inc_or_dec = inc_or_dec
        self.reset()
        

    def reset(self):
        self.val = []
        self.epochs = []
        self.count = 0
        self.best_val = float("inf") if self.inc_or_dec=='decaying' else float("-inf")
        self.best_count = 0
        self.best_epoch = 0

    def update(self, val, epoch):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val.append(val)
        self.epochs.append(epoch)
        
        if (self.inc_or_dec=='decaying' and val < self.best_val) or (self.inc_or_dec=='increasing' and val > self.best_val):
            self.best_val = val
            self.best_count = self.count
            self.best_epoch = epoch
        self.count += 1