import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import create_convblock1d, create_convblock2d

class PosPool_abs(nn.Module):
    def __init__(self, in_channels, mode=1, norm_args=None, act_args=None):
        """A PosPool operator for local aggregation

        Args:
            in_channels: input channels.
        """
        super(PosPool_abs, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        
        if mode==1:
            self.conv =  nn.Sequential(create_convblock1d(3, 3, norm_args=norm_args, act_args=act_args), create_convblock1d(3, in_channels, norm_args=None, act_args=None))
        elif mode==2:
            self.conv21 = create_convblock1d(in_channels, 3*in_channels, norm_args=None, act_args=act_args)
            self.conv22 = create_convblock1d(in_channels*3, in_channels, norm_args=None, act_args=None)
        elif mode==3:
            self.conv31 = create_convblock1d(in_channels, 3*in_channels, norm_args=None, act_args=act_args)
            self.conv32 = create_convblock1d(in_channels*3, in_channels, norm_args=None, act_args=None)
        elif mode==4:
            self.conv4 = create_convblock1d(in_channels*3, in_channels, norm_args=norm_args, act_args=act_args)

    def forward(self, xyz, feature):
        """
        Args:
            xyz: [B, N, 3]
            feature: [b, c, n]

        Returns:
           position: [B, C_out, 3]
        """
        B = xyz.shape[0]
        C = self.in_channels
        npoint = xyz.shape[1]
        xyz_new = xyz.transpose(1,2).contiguous()#b,3,n
        if self.mode == 1:
           feature_new = feature + self.conv(xyz_new)
        elif self.mode ==2:
            position_embedding = torch.unsqueeze(xyz_new, 1)
            feature_new = self.conv21(feature)
            feature_new = feature_new.view(B, C , 3, npoint)
            feature_new = position_embedding * feature_new  # (B, C//3, 3, npoint, nsample)
            feature_new = feature_new.view(B, C*3, npoint)  # (B, C, npoint, nsample)
            feature_new = self.conv22(feature_new)
        elif self.mode==3:
            feat_dim = C // 2
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz_new.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * xyz_new, -1)  # (B, 3, npoint, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 3, 2).contiguous()
            position_embedding = position_embedding.view(B, C*3, npoint)  # (B, 3*C, npoint)
            feature_new = self.conv31(feature)
            feature_new = feature_new * position_embedding  # (B, C, npoint)
            #feature_new = feature_new + position_embedding  # (B, C, npoint)
            feature_new = self.conv32(feature_new)
        elif self.mode==4:
            feat_dim = C // 2
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz_new.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * xyz_new, -1)  # (B, 3, npoint, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 3, 2).contiguous()
            position_embedding = position_embedding.view(B, C*3, npoint)  # (B, 3*C, npoint)
            position_embedding = self.conv4(position_embedding)
            #feature_new = feature * position_embedding  # (B, C, npoint)
            feature_new = feature + position_embedding  # (B, C, npoint)

        return feature_new

class PosPool_rel(nn.Module):
    def __init__(self, in_channels, mode=1, norm_args=None, act_args=None):
        """A PosPool operator for local aggregation

        Args:
            in_channels: input channels.
        """
        super(PosPool_rel, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        
        if mode==1:
            self.conv =  nn.Sequential(create_convblock2d(3, 3, norm_args=None, act_args=act_args), create_convblock2d(3, in_channels, norm_args=None, act_args=None))
        elif mode==2:
            self.conv21 = create_convblock2d(in_channels, 3*in_channels, norm_args=None, act_args=act_args)
            self.conv22 = create_convblock2d(in_channels*3, in_channels, norm_args=None, act_args=None)
        elif mode==3:
            self.conv31 = create_convblock2d(in_channels, 3*in_channels, norm_args=None, act_args=act_args)
            self.conv32 = create_convblock2d(in_channels*3, in_channels, norm_args=None, act_args=None)
        elif mode==4:
            self.conv4 = create_convblock2d(in_channels*3, in_channels, norm_args=norm_args, act_args=act_args)

    def forward(self, xyz, feature):
        """
        Args:
            xyz: [B, 3, N, k]
            feature: [b, c, n, k]

        Returns:
           position: [B, C_out, 3]
        """
        B = xyz.shape[0]
        C = self.in_channels
        npoint = feature.shape[2]
        k = feature.shape[-1]

        if self.mode == 1:
           feature_new = feature + self.conv(xyz)
        elif self.mode ==2:
            position_embedding = torch.unsqueeze(xyz, 1)
            feature_new = self.conv21(feature)
            feature_new = feature_new.reshape(B, C , 3, npoint, k).contiguous()
            feature_new = position_embedding * feature_new  # (B, C//3, 3, npoint, nsample)
            feature_new = feature_new.reshape(B, C*3, npoint, k).contiguous()  # (B, C, npoint, nsample)
            feature_new = self.conv22(feature_new)
        elif self.mode==3:
            feat_dim = C // 2
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * xyz, -1)  # (B, 3, npoint, k, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, k, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, k, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, k, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, k, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.reshape(B, C*3, npoint, k)  # (B, 3*C, npoint, k)
            feature_new = self.conv31(feature)
            feature_new = feature_new * position_embedding  # (B, C, npoint)
            #feature_new = feature_new + position_embedding  # (B, C, npoint)
            feature_new = self.conv32(feature_new)
        elif self.mode==4:
            feat_dim = C // 2
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * xyz, -1)  # (B, 3, npoint, k, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, k, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, k, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, k, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, k, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.reshape(B, C*3, npoint, k).contiguous()  # (B, 3*C, npoint, k)
            position_embedding = self.conv4(position_embedding)
            #feature_new = feature * position_embedding  # (B, C, npoint)
            feature_new = feature + position_embedding  # (B, C, npoint)

        return feature_new