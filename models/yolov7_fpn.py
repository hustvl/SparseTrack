import torch
import torch.nn as nn
from .yolov7_backbone import Yolov7_backbone, SPPCSPC, Conv, Elan, DownC

class Yolov7_fpn(nn.Module):
  def __init__(self, in_ch = [320, 640, 1280, 1280], out_ch = [320, 640, 1280]):
    super().__init__()
    self.backbone = Yolov7_backbone(ch_in=3, ch_out=in_ch)
    self.spp = SPPCSPC(in_ch[3], 640)

    self.cv0 = Conv(640, 320, 1, 1)
    self.up0 = nn.UpsamplingNearest2d(scale_factor = 2)
    self.in_cv0 = Conv(in_ch[2], 320, 1, 1)
    self.elan0 = Elan(640, 320, 256)

    self.cv1 = Conv(320, 160, 1, 1)
    self.up1 = nn.UpsamplingNearest2d(scale_factor = 2)
    self.in_cv1 = Conv(in_ch[1], 160, 1, 1)
    self.elan1 = Elan(320, 160, 128)

    self.down2 = DownC(160, 320)
    self.elan2 = Elan(640, 320, 256)

    self.down3 = DownC(320, 640)
    self.elan3 = Elan(1280, 640, 512)

    # out conv
    self.out_p3 = Conv(160, out_ch[0], 3, 1)
    self.out_p4 = Conv(320, out_ch[1], 3, 1)
    self.out_p5 = Conv(640, out_ch[2], 3, 1)

  def forward(self, x):
    feat_list=self.backbone(x)

    spp_out=self.spp(feat_list[-1])

    f1=self.up0(self.cv0(spp_out))
    f1=torch.cat([self.in_cv0(feat_list[-2]), f1], 1)
    f1=self.elan0(f1)

    f2=self.up1(self.cv1(f1))
    f2=torch.cat([self.in_cv1(feat_list[-3]), f2], 1)
    f2=self.elan1(f2)

    d2=self.down2(f2)
    d2=torch.cat([d2, f1], 1)
    d2=self.elan2(d2)

    d3=self.down3(d2)
    d3=torch.cat([d3, spp_out], 1)
    d3=self.elan3(d3)

    #out conv
    out0=self.out_p3(f2)
    out1=self.out_p4(d2)
    out2=self.out_p5(d3)

    return [out0, out1, out2] # /8 /16 /32 
