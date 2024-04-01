from re import X
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import create_convblock1d, create_convblock2d


class SELayer_global(nn.Module):
    '''
    x:b, c, n
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None):
        super(SELayer_global, self).__init__()
        mid_channels = in_channels // ration

        self.fc = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
        #self.fc2 = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
        self.sig = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()

    def forward(self, x):
        '''
        x:b, c, n
        '''
        y_mean = torch.mean(x, -1, keepdim=True)
        y_max = torch.max(x, -1, keepdim=True)[0]
        avg_out = self.fc(y_mean)
        max_out = self.fc(y_max)
        #out = avg_out*0.2 + max_out
        out = avg_out + max_out
        out = self.sig(out)
        out = out*x +x
        
        return out

class SELayer_local(nn.Module):
    '''
    x:b, c, n, k
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None):
        super(SELayer_local, self).__init__()
        mid_channels = in_channels // ration

        self.fc = nn.Sequential(create_convblock2d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock2d(mid_channels, in_channels, norm_args=None, act_args=None))
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        x:b, c, n, k
        '''
        y_mean = torch.mean(x, -1, keepdim=True)
        y_max = torch.max(x, -1, keepdim=True)[0]
        avg_out = self.fc(y_mean)
        max_out = self.fc(y_max)
        out = avg_out*0.2 + max_out
        #out = self.sigmoid(out)

        out = self.softmax(out)
        #out = out*x +x
        
        return out*x

class pt_global(nn.Module):
    '''
    x:b, c, n
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None):
        super(pt_global, self).__init__()
        mid_channels = in_channels // ration
        self.linear_q = create_convblock1d(in_channels, in_channels, norm_args=None, act_args=act_args)
        self.linear_k = create_convblock1d(in_channels, in_channels, norm_args=None, act_args=act_args)
        self.linear_v = create_convblock1d(in_channels, in_channels, norm_args=None, act_args=act_args)

        self.pos_encoding =  nn.Sequential(create_convblock1d(3, 3, norm_args=None, act_args=act_args), create_convblock1d(3, in_channels, norm_args=None, act_args=None))
        
        self.linear_w = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, p=None):
        '''
        x:b, c, n
        '''
        if p==None:
            x_q = torch.max(x, -1, keepdim=True)[0]
            w = self.linear_k(x) - self.linear_q(x_q)
            w = self.softmax(self.linear_w(w))
            out= w*x+ x
        else:
            pos_embedding = self.pos_encoding(p.transpose(1,2).contiguous())
            x = x+pos_embedding
            w = self.linear_k(x) - self.linear_q(x) 
            w = self.softmax(self.linear_w(w))
            out= w*x + x
        
        return out

class pt_local(nn.Module):
    '''
    x:b, c, n, k
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None):
        super(pt_local, self).__init__()
        mid_channels = in_channels // ration
        self.linear_q = create_convblock2d(in_channels, in_channels, norm_args=None, act_args=act_args)
        self.linear_k = create_convblock2d(in_channels, in_channels, norm_args=None, act_args=act_args)
        self.linear_v = create_convblock2d(in_channels, in_channels, norm_args=None, act_args=act_args)

        self.pos_encoding =  nn.Sequential(create_convblock2d(3, 3, norm_args=None, act_args=act_args), create_convblock2d(3, in_channels, norm_args=None, act_args=None))
        
        self.linear_w = nn.Sequential(create_convblock2d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock2d(mid_channels, in_channels, norm_args=None, act_args=None))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, p=None):
        '''
        x:b, c, n, k
        '''
        if p==None:
            w = self.linear_k(x) - self.linear_q(x)
            w = self.softmax(self.linear_w(w))
            out= w*self.linear_v(x)+ x
        else:
            pos_embedding = self.conv(p)
            w = self.linear_k(x) - self.linear_q(x) + pos_embedding
            w = self.softmax(self.linear_w(w) + pos_embedding)
            out= w*self.linear_v(x)
        
        return out

class GCLayer_global(nn.Module):
    '''
    x:b, c, n
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None):
        super(GCLayer_global, self).__init__()
        mid_channels = in_channels // ration

        self.fc = create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args)
        self.fc1 = create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None)
        self.fc2 = create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None)

        #self.fc = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
        #self.fc1 = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x:b, c, n
        '''
        y_mean = torch.mean(x, -1, keepdim=True)
        y_max = torch.max(x, -1, keepdim=True)[0]
        #out = torch.cat((y_max, y_mean), -1)
        y_mean = self.fc1(self.fc(y_mean))
        y_max = self.fc2(self.fc(y_max))
        
        out = y_mean + y_max
        out = self.sigmoid(out)
        #out = self.sigmoid(out)
        out = out*x +x
        
        return out

class GCLayer_local(nn.Module):
    '''
    x:b, c, n, k
    '''
    def __init__(self, in_channels, ration=4, norm_args=None, act_args=None, mode =1):
        super(GCLayer_local, self).__init__()
        mid_channels = in_channels // ration
        self.mode = mode

        if mode==1:
            self.pool = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, 1, norm_args=None, act_args=None))
            self.softmax =  nn.Softmax(dim=-1)
            self.fc = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
            self.sigmoid = nn.Sigmoid()
        elif mode==2:
            self.pool = nn.Sequential(create_convblock2d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock2d(mid_channels, 1, norm_args=None, act_args=None))
            self.softmax =  nn.Softmax(dim=-1)
            self.fc = nn.Sequential(create_convblock1d(in_channels, mid_channels, norm_args=None, act_args=act_args), create_convblock1d(mid_channels, in_channels, norm_args=None, act_args=None))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x:b, c, n, k
        '''
        if self.mode==1:
            b, c, n, k = x.shape
            y = x.view(b, c, -1).contiguous()#b, c, n*k
            y_pool = self.pool(y) #b, 1, n*k
            y_pool = self.softmax(y_pool)#b, 1, n*k
            out = torch.matmul(y, y_pool.transpose(1,2).contiguous())# b, c, 1
            out = self.fc(out)
            out = self.sigmoid(out)
            out = (out.unsqueeze(-1))*x +x
        elif self.mode==2:
            y_pool = self.pool(x) #b,1,n,k
            y_pool = self.softmax(y_pool) #b, 1, n, k
            out = torch.matmul(x.transpose(1,2).contiguous(), y_pool.permute(0, 2, 3, 1).contiguous()).transpose(1, 2).contiguous()
            out = self.fc(out)
            out = self.sigmoid(out)
            out = out*x + x
        
        return out