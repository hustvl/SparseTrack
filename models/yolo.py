import torch, logging, thop
import torch.nn as nn
from .baseblock import C2f, Conv, SPPF, Concat, RepConv,RepC3, DWConv,ConvTranspose
from .head import Detect
from .loss import v8DetectionLoss
from utils.model_utils import fuse_conv_and_bn, fuse_deconv_and_bn, yaml_model_load, parse_model, intersect_dicts, initialize_weights, time_sync, model_info, scale_img
from utils.ops import non_max_suppression
from copy import deepcopy
from detectron2.structures import Instances, Boxes
LOGGER = logging.getLogger("detectron2")     
        
class DetectionModel(nn.Module):
    """YOLOv8 detection model."""
    def __init__(self, 
                 cfg='yolov8n.yaml',  ch=3,  nc=None,  cls_idx = [0,], 
                 conf=0.1, iou=0.7, agnostic=False, multi_label=False, 
                 num_max_dets=1000, verbose=True
    ):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist--save layer-idx
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)
        
        # Build strides
        m = self.model[-1]  # Detect()
        if 'yolov8' in cfg:
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.predict(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
        
        # infer params
        self.conf = conf
        self.iou = iou
        self.classes = cls_idx # A list of class indices to consider. If None, all classes will be considered.
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = num_max_dets  
        self.device = None

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()

    def init_criterion(self):
        return v8DetectionLoss(self)
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x: mix_img, padded_labels, img_info, np.array([idx])

        Returns:
            (torch.Tensor): The output of the network.
        """
        # import pdb;pdb.set_trace()
        if self.training:  # for cases of training and validating while training.
            batch_input = {}
            batch_input['img'] = x[0].to(self.device)
            batch_input['anns'] = x[1].to(self.device) # cls cx cy w h -1
            return self.loss(batch_input, *args, **kwargs)
        else:
            return self.infer(x, *args, **kwargs)
        
    def infer(self, x, profile=False):
        def _batch_data_prerocess(batch_data):
            ori_image_sizes, input_image_sizes, input_images= [], [], []
            for data_per_img in batch_data:
                ori_image_sizes.append([data_per_img['height'], data_per_img['width']])
                input_image_sizes.append([data_per_img['image'].shape[1], data_per_img['image'].shape[2]])
                input_images.append(data_per_img['image'])
            return ori_image_sizes, input_image_sizes, input_images
        
        # preprocess
        ori_image_sizes, input_image_sizes, input_images =_batch_data_prerocess(x)
        # Network forward
        batch_ouput = self.predict(torch.stack(input_images).to(self.device), profile)
        # NMS
        batch_ouput = non_max_suppression(batch_ouput[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  
        
        results = []
        # batch_ouput ordered as (x1, y1, x2, y2, obj_conf, class_pred)
        for  output_per_image, input_img_size, ori_img_size in zip(
            batch_ouput, input_image_sizes, ori_image_sizes
        ):
            result = Instances(ori_img_size)
            pred_bboxes = output_per_image[:, :4]
            scale = min(input_img_size[0] / float(ori_img_size[0]), input_img_size[1] / float(ori_img_size[1]))
            pred_bboxes /= scale
            result.pred_boxes = Boxes(pred_bboxes)
            result.scores = output_per_image[:, 4]
            result.pred_classes = output_per_image[:, 5]
            results.append({"instances": result})
            # detections preprocess
        return results
    
    def predict(self, x, profile=False, extract = False):
        """
        Perform a forward pass through the network. without NMS

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save outputs from the specified layers
        
        if extract:
            return x, y, self.save
        else:
            return x
    
    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=[x.clone() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.clone() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
            
    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.--- used for inference

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if hasattr(m, 'conv') and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if hasattr(m, 'conv_transpose') and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                # if isinstance(m, RepConv):
                #     m.fuse_convs()
                #     m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self
    
    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers. --- used for inference

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
        
    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    
    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.predict(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)
            
    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
        return self