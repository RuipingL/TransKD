from this import d
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from models.Segformer import mit_b0,mit_b1,mit_b2#,mit_b3,mit_b4,mit_b5
class SK(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse, len=32, reduce=16):
        super(SK, self).__init__()
        len = max(mid_channel // reduce, len)
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            #https://github.com/syt2/SKNet
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(len),
                nn.ReLU(inplace=True)
            )
            self.fc1 = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fcs = nn.ModuleList([])
            for i in range(2):
                self.fcs.append(
                    nn.Conv2d(len, mid_channel, kernel_size=1, stride=1)
                )
            self.softmax = nn.Softmax(dim=1)
        # else:
        #     self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        # b,c,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.fuse:
            # upsample residual features
            shape = x.shape[-2:]
            b = x.shape[0]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            feas_U = [x,y]
            feas_U = torch.stack(feas_U,dim=1)
            attention = torch.sum(feas_U, dim=1)
            attention = self.gap(attention)
            if b ==1:
                attention = self.fc1(attention)
            else:
                attention = self.fc(attention)
            attention = [fc(attention) for fc in self.fcs]
            attention = torch.stack(attention, dim=1)
            attention = self.softmax(attention)
            x = torch.sum(feas_U * attention, dim=1)

        # output 
        y = self.conv2(x)
        return y, x

class SKF(nn.Module):
    def __init__(
        self,student, in_channels, out_channels, mid_channel, embed, add, num_heads=[1,2,5,8]
    ):
        super(SKF, self).__init__()
        self.student = student
        self.add = add

        skfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            skfs.append(SK(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.skfs = skfs[::-1]
        self.embed = embed
        self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
        self.embed2_linearproject = nn.Linear(in_channels[1], out_channels[1])
        self.embed3_linearproject = nn.Linear(in_channels[2], out_channels[2])
        self.embed4_linearproject = nn.Linear(in_channels[3], out_channels[3])


        self.attn4 = Attention(dim=in_channels[3],num_heads=num_heads[3])        
        ##################################################################################
        #Lite Transformer https://arxiv.org/abs/2004.11886
        #GLU https://arxiv.org/abs/1612.08083
        self.conv1 = nn.Conv1d(in_channels[3],in_channels[3],kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv1d(in_channels[3],in_channels[3],kernel_size=3,stride=1,padding=1)
        self.sig = nn.Sigmoid()
        self.atten_weight=nn.Parameter(torch.FloatTensor(1))
        self.conv_weight=nn.Parameter(torch.FloatTensor(1))




    def forward(self, x):
        student_features = self.student(x,is_feat=True)
        embed = student_features[2]
        logit = student_features[1]
        x = student_features[0][::-1]
        results = []
        embedproj = []
        out_features, res_features = self.skfs[0](x[0])
        results.append(out_features)
        for features, skf in zip(x[1:], self.skfs[1:]):
            out_features, res_features = skf(features, res_features)
            results.insert(0, out_features) 
        embed3 = embed[3].transpose(1,2).contiguous()

        embed3_1 = self.conv1(embed3)
        embed3_2 = self.sig(self.conv2(embed3))
        embed3_conv = embed3_1*embed3_2
        embed3_conv = embed3_conv.transpose(1,2).contiguous()
        embed3 = self.atten_weight*self.attn4(embed[3])+self.conv_weight*embed3_conv
        if self.add:
            embed3 = embed3+embed[3]
        embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
        embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
        embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
        embedproj = [*embedproj, self.embed4_linearproject(embed3)]

        return results, logit, embedproj


def build_kd_trans(model,embed, add):
    in_channels = [32, 64, 160, 256]
    out_channels = [64, 128, 320, 512]
    mid_channel = 64
    student = model
    model = SKF(student, in_channels, out_channels, mid_channel,embed, add)
    return model

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap



class CriterionCWD(nn.Module):

    def __init__(self,norm_type='none',divergence='mse',temperature=1.0):
    
        super(CriterionCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

        
        

    def forward(self,preds_S, preds_T):
        
        n,c,h,w = preds_S.shape
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)
#################################################################################################################################3
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
