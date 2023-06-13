import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
  # Pad to 'same'
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Conv(nn.Module):
    # Standard convolution
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super(Conv, self).__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

  def fuseforward(self, x):
    return self.act(self.conv(x))

class DownC(nn.Module):
  # Spatial pyramid pooling layer used in YOLOv3-SPP
  def __init__(self, c1, c2, n=1, k=2):
    super(DownC, self).__init__()
    c_ = int(c2 / 2)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_, c2//2, 3, k)
    self.cv3 = Conv(c1, c2//2, 1, 1)
    self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

  def forward(self, x):
    return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)

class Elan(nn.Module):
  def __init__(self, c1, c2, hide_ch = None):
    super().__init__()
    if hide_ch == None:
      hide_ch = int(c2 / 5)
    self.c_10 = Conv(c1, hide_ch, 1, 1)
    self.c_2 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_3 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_4 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_5 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_6 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_7 = Conv(hide_ch, hide_ch, 3, 1)
    self.c_11 = Conv(c1, hide_ch, 1, 1)
    self.fuse = Conv(5*hide_ch, c2, 1, 1)

  def forward(self, x):
    x0=self.c_11(x)
    x1=self.c_10(x)
    x2=self.c_3(self.c_2(x1))
    x3=self.c_5(self.c_4(x2))
    x4=self.c_7(self.c_6(x3))
    out=self.fuse(torch.cat([x0, x1, x2, x3, x4], 1))
    return out

class SPPCSPC(nn.Module):
  # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
    super(SPPCSPC, self).__init__()
    c_ = int(2 * c2 * e)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c1, c_, 1, 1)
    self.cv3 = Conv(c_, c_, 3, 1)
    self.cv4 = Conv(c_, c_, 1, 1)
    self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    self.cv5 = Conv(4 * c_, c_, 1, 1)
    self.cv6 = Conv(c_, c_, 3, 1)
    self.cv7 = Conv(2 * c_, c2, 1, 1)

  def forward(self, x):
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
    y2 = self.cv2(x)
    return self.cv7(torch.cat((y1, y2), dim=1))
  

class Yolov7_backbone(nn.Module):
  def __init__(self, ch_in=3, ch_out=[320, 640, 1280, 1280], hide_base_ch=40):
    super().__init__()
    self.channels = [80, 160] + ch_out
    self.input_layer = Conv(ch_in, hide_base_ch, 3, 1)

    self.level0 = Conv( hide_base_ch,  hide_base_ch * 2, 3, 2)

    self.level1 = Conv(hide_base_ch * 2, hide_base_ch * 2, 3, 1)

    self.level2 = nn.Sequential(
      Conv(hide_base_ch * 2, hide_base_ch * 4, 3, 2),
      Elan(hide_base_ch * 4, ch_out[0]),
      )#---p2 /4 -- 320

    self.level3 = nn.Sequential(
      DownC(ch_out[0], ch_out[0]),
      Elan(ch_out[0], ch_out[1])
    )# ---p3 /8 -- 640

    self.level4 = nn.Sequential(
      DownC(ch_out[1], ch_out[1]),
      Elan(ch_out[1], ch_out[2])
    )# ---p4 /16 -- 1280

    self.level5 = nn.Sequential(
      DownC(ch_out[2], ch_out[2]),
      Elan(ch_out[2], ch_out[3])
    )# ---p5 /32 -- 1280

  def forward(self, x):
    y = [] 
    x = self.input_layer(x)
    for i in range(6):
      x = getattr(self, 'level{}'.format(i))(x)
      y.append(x)
    return y