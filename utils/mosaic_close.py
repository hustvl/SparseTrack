import logging
from detectron2.engine.train_loop import HookBase

class MosaicClose(HookBase):
    "close mosaic augmentation before the last X epochs"
    def __init__(self, trainer, iters_per_eps, no_aug_eps, is_distributed):
        self._trainer = trainer
        self.iters_per_eps = iters_per_eps
        self.no_aug_eps = no_aug_eps
        self.is_distributed = is_distributed
        self._logger = logging.getLogger("detectron2.utils.events")
        
    def before_step(self):
        if self._trainer.iter + 1 == self._trainer.max_iter - self.no_aug_eps * self.iters_per_eps:
            self._logger.info("--->No mosaic aug now!")
            self._trainer.data_loader.dataset.enable_mosaic = False
            self._trainer.data_loader.close_mosaic()
            # self._logger.info("--->Add additional L1 loss now!") 为什么不用L1损失的原因是 ： 容易对数据集过拟合
            # if self.is_distributed:
            #     self._trainer.model.module.head.use_l1 = True
            # else:
            #     self._trainer.model.head.use_l1 = True
    