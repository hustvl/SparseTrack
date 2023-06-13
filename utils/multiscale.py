import logging, torch
from detectron2.engine.train_loop import HookBase
from detectron2.utils import comm
import random
import torch.distributed as dist

class MultiScale(HookBase):
    """
    Multi-scaled training hook
    """
    def __init__(self, trainer, input_size, random_size, log_period, is_distributed):
        # import pdb;pdb.set_trace()
        self._trainer = trainer
        self.input_size = input_size
        self.ori_size = input_size
        self.random_size = random_size
        self.print_period = log_period
        self.is_or_distributed = is_distributed
        self._logger = logging.getLogger("detectron2.utils.events")

    def after_step(self):
        # after iter, printing current input size 
        if (self._trainer.iter + 1) % self.print_period == 0 or (
            self._trainer.iter == self._trainer.max_iter - 1
        ):
            self._logger.info(f" The current input size is: {self.input_size},  The original input size is: {self.ori_size} !")
            
        # random resizing
        if self.random_size is not None and (self._trainer.iter + 1) % 10 == 0:
            self.input_size = self._random_resize(
                self._trainer.data_loader, self.is_or_distributed
            )
 
    def _random_resize(self, data_loader, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if comm.is_main_process():
            size_factor = self.ori_size[1] * 1.0 / self.ori_size[0]
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = data_loader.change_input_dim(
            multiple=(tensor[0].item(), tensor[1].item()), random_range=None
        )
        return input_size