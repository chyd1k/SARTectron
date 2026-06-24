from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
import numpy as np
import torch
import time
import datetime
import logging
import sys
import itertools
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

def set_StreamHandler():
    if not comm.is_main_process():
        return
    
    logger = logging.getLogger(__name__)
    h = logging.StreamHandler(sys.stdout)
    abbrev_name = "d2" if __name__ == "detectron2" else __name__
    h.flush = sys.stdout.flush
    h.setFormatter(_ColorfulFormatter(
                    colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                    datefmt="%m/%d %H:%M:%S",
                    root_name=__name__,
                    abbrev_name=str(abbrev_name),
                ))
    logger.addHandler(h)


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        # if comm.is_main_process():
        #     set_StreamHandler()

    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        # forward без DDP-обёртки -> на каждом шаге нет коллективов,
        # поэтому неравный размер шарда (391 vs 390) больше не вешает обучение
        model = self._model.module if hasattr(self._model, "module") else self._model

        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)),
                    n=5,
                )
            losses.append(self._get_loss(inputs, model))

        # все ранги вызывают одни и те же коллективы в одном порядке
        comm.synchronize()
        all_losses = comm.gather(losses, dst=0)
        if comm.is_main_process():
            all_losses = list(itertools.chain(*all_losses))
            self.trainer.storage.put_scalar("validation_loss", np.mean(all_losses))
        comm.synchronize()
        return losses

    def _get_loss(self, data, model):
        metrics_dict = model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return sum(metrics_dict.values())

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
