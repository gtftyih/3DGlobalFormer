import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp
import torch_dct as dct

from timm.layers import DropPath


class Model(nn.Module):
    def __init__(self, args=None, in_dim=2, out_dim=3):
        super().__init__()

        if args == None:
            layers, d_hid, frames=6, 256, 27
            num_joints_in, num_joints_out = 17, 17
        else:
            layers, d_hid, frames = args.layers, args.d_hid, args.frames
            num_joints_in, num_joints_out = args.n_joints, args.out_joints

        self.pose_emb_s = nn.Linear(2*num_joints_in, d_hid//2, bias=False)
        self.LP = 9
        self.pose_emb_t = nn.Linear(2*self.LP, d_hid//2, bias=False)
        self.pose_emb_0 = nn.Linear(2, d_hid, bias=False)
        self.gelu = nn.GELU()

        self.globalformer = GlobalFormer(layers, frames, num_joints_in, d_hid)
        self.regress_head = nn.Linear(d_hid, 3, bias=False)


    def forward(self, x, pre_mask=False, mask=None, spatial_mask=None, is_3dhp=False):
        # dimension tranfer
        if is_3dhp:
            x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()  # B,T,J,2,1 (for 3dhp)

        b, t, s, c = x.shape  #batch,frame,joint,coordinate
        p_s, p_t = [], []
        for i in range(s):
            p_s.append(torch.roll(x, i, dims=2))
        p_s = torch.cat(p_s, dim=-1)
        x_t = F.pad(x.transpose(1, 2),(0,0,(self.LP-1)//2,(self.LP-1)//2), 'constant', 0) # b, s, t+l, c
        for j in range(self.LP):
            p_t.append(torch.roll(x_t, (self.LP-1)//2-j, dims=2))
        p_t = torch.cat(p_t, dim=-1)
        p_t = p_t[:,:,((self.LP-1)//2):(1-self.LP)//2,:].transpose(1, 2)
        x_s = self.pose_emb_s(p_s)
        x_t = self.pose_emb_t(p_t)
        x_0 = self.pose_emb_0(x)
        x = 1e-4*torch.cat((x_s,x_t),-1)+x_0
        x = self.gelu(x)
        # spatio-temporal correlation
        x = self.globalformer(x)
        # regression head
        x = self.regress_head(x)

        return x


class FreqSH(nn.Module):
    def __init__(self, channel_ratio, act_layer=nn.GELU, input_dim=128, freq_ratio=4, LP=None, drop=0.):
        super().__init__()
        
        self.LP=LP         
        self.freq_dim = input_dim//freq_ratio
        if self.LP:
            self.propad = nn.Linear(self.LP, channel_ratio)
            channel_ratio = LP      
        self.freq_embedding = nn.Linear(input_dim, self.freq_dim)
        channel = channel_ratio * self.freq_dim
        self.down1 = nn.Linear(channel, channel//2)
        self.down2 = nn.Linear(channel//2, channel//4)
        self.downN = nn.Linear(channel//4, channel//4)
        self.up2 = nn.Linear(channel//4, channel//2)
        self.up1 = nn.Linear(channel//2, channel)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.unfreq_embedding = nn.Linear(self.freq_dim, input_dim)


    def forward(self, input):
        b, i, j, c = input.shape
        input = self.freq_embedding(input).permute(0, 1, 3, 2) #b,i,c,j
        input = dct.dct(input).permute(0,1,3,2).contiguous()
        if self.LP:
            input = input[:,:,:self.LP,:] #b, i, LP, c
        input = input.view(b,i,-1)
        down1 = self.down1(input)       
        down2 = self.down2(self.drop(self.act(down1)))
        downN = self.downN(self.drop(self.act(down2)))
        up2 = self.up2(downN)+down1
        up1 = self.up1(up2)+input
        output = up1.view(b,i,-1,self.freq_dim)
        if self.LP:
            output = output.permute(0,3,1,2)
            output = self.propad(output).permute(0,2,3,1).contiguous()
        output = dct.idct(output.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        output = self.unfreq_embedding(output)
        return output


class Spatial_Attention(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()        
        self.head = head
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.layer_norm = nn.LayerNorm(d_coor)
        self.scale = (d_coor) ** -0.5
        self.conv2d = nn.Conv2d(d_coor, d_coor, kernel_size=3, stride=1, padding=1, groups=d_coor)
        self.pool2d = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.freqmlp = FreqSH(channel_ratio=d_joint, input_dim=d_coor, freq_ratio=4, drop=0.)  
        self.drop = DropPath(0.5)      

    def forward(self, x):
        b, t, s, c = x.shape
        x_dct = x.clone() #dct_s
        x_dct = self.drop(self.freqmlp(x_dct)).view(b, t, s, -1) #b,t,s,c
        qkv = self.qkv(x).reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q = rearrange(q, 'b t s (h c) -> (b h t) s c', h=self.head) 
        k = rearrange(k, 'b t s (h c) -> (b h t) c s', h=self.head)  # b,t,s,c-> b*h*t,c//h,s
        att = ((q @ k) * self.scale).softmax(-1)  # b*h*t,s,s
        v= rearrange(v, 'b t s c -> b c t s ')
        HDPE = self.conv2d(v) # b,c,t,s
        LDPE = self.pool2d(HDPE) # b,c,t,s
        HDPE = rearrange(HDPE, 'b (h c) t s  -> (b h t) s c ', h=self.head)
        LDPE = rearrange(LDPE, 'b (h c) t s  -> (b h t) s c ', h=self.head)
        v = rearrange(v, 'b (h c) t s -> (b h t) s c', h=self.head)  # b*h*t,s,c//h
        h = att @ v + HDPE + 1e-3*self.drop(LDPE)        # b*h*t,s,c//2//h 
        h = rearrange(h, '(b h t) s c -> b t s (h c)', h=self.head, t=t, s=s)   # b*h*t,s,c//h -> # b,t,s,c ##  
        return h+x_dct


class Temperol_Attention(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()        
        self.head = head
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.layer_norm = nn.LayerNorm(d_coor)
        self.scale = (d_coor) ** -0.5
        self.conv2d = nn.Conv2d(d_coor, d_coor, kernel_size=3, stride=1, padding=1, groups=d_coor)
        self.pool2d = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.drop = DropPath(0.5)   
        self.LP = 9
        self.freqmlp = FreqSH(channel_ratio=d_time, input_dim=d_coor, freq_ratio=4, LP=self.LP, drop=0.) 

    def forward(self, x):
        b, t, s, c = x.shape
        x_dct = x.clone().transpose(1, 2) #b s t c
        x_dct = self.drop(self.freqmlp(x_dct)).view(b, s, t, -1) #b,s,t,c
        x_dct = x_dct.transpose(1, 2) #b,t,s,c//2
        qkv = self.qkv(x).reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q = rearrange(q, 'b t s (h c) -> (b h s) t c', h=self.head)  # b,t,s,c-> b*h*s,t,c//h
        k = rearrange(k, 'b t s (h c) -> (b h s) c t', h=self.head)  # b,t,s,c-> b*h*s,c//h,t
        att = ((q @ k) * self.scale).softmax(-1)  # b*h*s,t,t
        v= rearrange(v, 'b t s c -> b c t s ')
        HDPE = self.conv2d(v) # b,c,t,s
        LDPE = self.pool2d(HDPE) # b,c,t,s
        HDPE = rearrange(HDPE, 'b (h c) t s  -> (b h s) t c ', h=self.head)
        LDPE = rearrange(LDPE, 'b (h c) t s  -> (b h s) t c ', h=self.head)
        v = rearrange(v, 'b (h c) t s -> (b h s) t c', h=self.head)  # b*h*s,t,c//h
        h = att @ v + HDPE + 1e-9*self.drop(LDPE)     # b*h*s,t,c//2//h 
        h = rearrange(h, '(b h s) t c -> b t s (h c)', h=self.head, t=t, s=s)  # b*h*t,s,c//h -> # b,t,s,c ##         
        return h+x_dct


class Freq_ATTENTION(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)
        self.emb_s = nn.Linear(d_coor, d_coor//2)
        self.emb_t = nn.Linear(d_coor, d_coor//2)
        self.proj = nn.Linear(d_coor, d_coor) 
        self.proj_s = nn.Linear(d_coor//2, d_coor) #
        self.proj_t = nn.Linear(d_coor//2, d_coor) #
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head

        self.freq_attention_s = Spatial_Attention(self.d_time, self.d_joint, d_coor//2, self.head)
        self.freq_attention_t = Temperol_Attention(self.d_time, self.d_joint, d_coor//2, self.head)
     
        #fusion
        self.fusion = nn.Linear(d_coor , 2)
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5) 


    def forward(self, input):
        b, t, s, c = input.shape

        h = input
        x = self.layer_norm(input)

        #x_s, x_t = x.chunk(2, 3)
        x_s = self.emb_s(x)
        x_t = self.emb_t(x)
        x_s = self.freq_attention_s(x_s)
        x_t = self.freq_attention_t(x_t)

        x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        alpha = self.fusion(x).softmax(dim=-1)
        x = self.proj_s(x_s) * alpha[..., 0:1] + self.proj_t(x_t) * alpha[..., 1:2]

        # projection and skip-connection
        x = self.proj(x)
        x = x + h
        return x


class Stacked_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)
        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)
        self.freq_att = Freq_ATTENTION(d_time, d_joint, d_coor)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.freq_att(input)
        x = x + self.drop(self.mlp(self.layer_norm(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFormer(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor ):
        super(GlobalFormer, self).__init__()

        self.num_block = num_block
        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.stacked_block = []
        for l in range(self.num_block):
            self.stacked_block.append(Stacked_BLOCK(self.d_time, self.d_joint, self.d_coor))
        self.stacked_block = nn.ModuleList(self.stacked_block)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.stacked_block[i](input)
        return input


if __name__ == "__main__":
    # inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Model(out_dim=3)
    inputs = torch.rand([1, 243, 17, 2])
    output = net(inputs)
    print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs,))
    print(flops)
    print(params)
