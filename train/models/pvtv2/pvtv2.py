from torch import nn
import torch.nn.functional as F
from .backbone import pvt_v2_b0, pvt_v2_b2
from .neck import FPN
from .decode_head import FPNHead

class PVTv2(nn.Module):
    def __init__(
        self,backbone,neck,decode_head
    ):
        super(PVTv2, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = decode_head


    def forward(self, x, is_feat=True):
        features, embeds = self.backbone(x)
        outs = self.neck(features)
        outs = self.head(outs)
        if is_feat:
            return features,outs,embeds
        else:
            return outs

def build_pvts(model,num_classes):
    if model == 'pvt_v2_b0':
        backbone = pvt_v2_b0()
        neck= FPN(in_channels=[32, 64, 160, 256])
        decode_head = FPNHead(num_classes=num_classes)
    if model == 'pvt_v2_b2':
        backbone = pvt_v2_b2()
        neck= FPN(in_channels=[64, 128, 320, 512])
        decode_head = FPNHead(num_classes=num_classes)
    model = PVTv2(backbone,neck,decode_head)
    return model
    
    