import torch
from torch import nn
import torch.nn.functional as F

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

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1) 

    def forward(self, x, y=None, shape=None):

        x = self.conv1(x)
        if self.fuse:
            shape = x.shape[-2:]
            b = x.shape[0]
            y = F.interpolate(y, shape, mode="nearest")

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
        self,student, in_channels, out_channels, mid_channel, embed
    ):
        super(SKF, self).__init__()
        self.student = student

        skfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            skfs.append(SK(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.skfs = skfs[::-1]
        self.embed = embed
        if self.embed == 5:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
            self.embed2_linearproject = nn.Linear(in_channels[1], out_channels[1])
            self.embed3_linearproject = nn.Linear(in_channels[2], out_channels[2])
            self.embed4_linearproject = nn.Linear(in_channels[3], out_channels[3])
        elif self.embed == 1:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
        elif self.embed == 2:
            self.embed1_linearproject = nn.Linear(in_channels[1], out_channels[1])
        elif self.embed == 3:
            self.embed1_linearproject = nn.Linear(in_channels[2], out_channels[2])
        elif self.embed == 4:
            self.embed1_linearproject = nn.Linear(in_channels[3], out_channels[3])


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

        if self.embed ==5:
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
            embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
            embedproj = [*embedproj, self.embed4_linearproject(embed[3])]
            return results, logit, embedproj
        elif self.embed == 0:
            return results, logit
        elif self.embed == 1:
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            return results, logit, embedproj
        elif self.embed == 2:
            embedproj = [*embedproj, self.embed1_linearproject(embed[1])]
            return results, logit, embedproj
        elif self.embed == 3:
            embedproj = [*embedproj, self.embed1_linearproject(embed[2])]
            return results, logit, embedproj
        elif self.embed == 4:
            embedproj = [*embedproj, self.embed1_linearproject(embed[3])]
            return results, logit, embedproj            
        else:
            assert 'the number of embeddings not supported'


def build_kd_trans(model,embed,in_channels = [32, 64, 160, 256], out_channels = [64, 128, 320, 512]):
    mid_channel = 64
    student = model
    model = SKF(student, in_channels, out_channels, mid_channel,embed)
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
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)
######################################################################################################################
class EmbedChannelNorm(nn.Module):
    def __init__(self):
        super(EmbedChannelNorm, self).__init__()
    def forward(self,embed):
        n,c,_ = embed.shape
        embed = embed.softmax(dim=-1)
        return embed



class CriterionEmbedCWD(nn.Module):

    def __init__(self,norm_type='none',divergence='mse',temperature=1.0):
    
        super(CriterionEmbedCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = EmbedChannelNorm()
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

        
        

    def forward(self,embed_S, embed_T):
        embed_S = embed_S.transpose(1, 2).contiguous()
        embed_T = embed_T.transpose(1, 2).contiguous()
        n,c,_ = embed_S.shape
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(embed_S/self.temperature)
            norm_t = self.normalize(embed_T.detach()/self.temperature)
        else:
            norm_s = embed_S[0]
            norm_t = embed_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c

        return loss * (self.temperature**2)

def hcl_feaw(fstudent, fteacher):
    loss_all = 0.0
    fea_weights = [0.1,0.1,0.5,1]
    for fs, ft,fea_w in zip(fstudent, fteacher,fea_weights):
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
        loss_all = loss_all + fea_w*loss
    return loss_all