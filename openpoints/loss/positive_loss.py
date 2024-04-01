import torch
import torch.nn as nn
import torch.nn.functional as F

class positive_loss(nn.Module):
    def __init__(self):
        super(positive_loss, self).__init__()
    
    def forward(self, pred, positive_site, num_cls):
        #print(pred.shape)
        out = F.softmax(pred, 1)
        loss_n = (1-positive_site)*(out - 1/num_cls)
        loss_n = torch.clamp(loss_n, min=0)
        loss_n = torch.gt(loss_n, 0)* (1-positive_site) * out
        return loss_n.mean()