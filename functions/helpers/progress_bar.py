from collections import OrderedDict
from numbers import Number
from tqdm import tqdm
from .meters import AverageMeter, RunningAverageMeter, TimeMeter


class ProgressBar:
    ''''
    Takes iterable like train_loader and functions exctly like this iterator if quiet is True. Otherwise it additionally provides a progress bar.
    '''
    def __init__(self, iterable, epoch, quiet=False):
        self.epoch = epoch
        self.quiet = quiet
        self.prefix = f"epoch {epoch:02d}"
        self.iterable = iterable if self.quiet else tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats, verbose=False):
        if not self.quiet:
            self.iterable.set_postfix(self.format_stats(stats, verbose), refresh=True)

    def format_stats(self, stats, verbose=False):
        postfix = OrderedDict(stats) # method set_postfix requires ordered_dict
        for key, value in postfix.items():
            if isinstance(value, Number):
                fmt = "{:.6f}" if value > 0.001 else "{:.3e}"
                postfix[key] = fmt.format(value)
            elif isinstance(value, AverageMeter) or isinstance(value, RunningAverageMeter):
                if verbose:
                    postfix[key] = f"{value.avg:.6f} ({value.val:.6f})"
                else:
                    postfix[key] = f"{value.avg:.6f}"
            elif isinstance(value, TimeMeter):
                postfix[key] = f"{value.elapsed_time:.1f}s"
            elif not isinstance(postfix[key], str):
                postfix[key] = str(value)
        return postfix

    def print(self, stats, verbose=False):
        postfix = " | ".join(key + " " + value.strip() for key, value in self.format_stats(stats, verbose).items())
        return f"{self.prefix + ' | ' if self.epoch is not None else ''}{postfix}"
    