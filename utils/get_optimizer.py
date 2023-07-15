import torch, logging
import torch.nn as nn
import torch.optim as optim
LOGGER = logging.getLogger("detectron2")

def get_optimizer(batch_size, basic_lr_per_img, model, momentum, weight_decay, warmup_epochs, warmup_lr_start):
 
    if warmup_epochs > 0:
        lr = warmup_lr_start
    else:
        lr = basic_lr_per_img * batch_size

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        # import pdb;pdb.set_trace()
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=lr, momentum=momentum, nesterov=True
    )
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": weight_decay}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    
    return optimizer