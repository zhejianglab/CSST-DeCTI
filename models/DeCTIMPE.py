import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath, trunc_normal_
from timm.layers import DropPath, trunc_normal_
from utils.tools import date_id_mapping
#######
# UNITS
#######
NUM_DATE=34
NUM_COLUMN = 64 
# NUM_COLUMN = 512 

class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, in_channels, in_length = x.size()
        
        # 检查通道数是否可被上采样因子整除
        assert in_channels % self.upscale_factor == 0, \
            f"Input channels must be divisible by upscale factor, got {in_channels} and {self.upscale_factor}"
        
        # 计算输出通道数
        out_channels = in_channels // self.upscale_factor
        out_length = in_length * self.upscale_factor
        
        # 重塑张量: [B, C, L] -> [B, C//r, r, L]
        x = x.view(batch_size, out_channels, self.upscale_factor, in_length)
        
        # 置换维度: [B, C//r, r, L] -> [B, C//r, L, r]
        x = x.permute(0, 1, 3, 2)
        
        # 合并最后两个维度: [B, C//r, L, r] -> [B, C//r, L*r]
        x = x.reshape(batch_size, out_channels, out_length)
        
        return x

class Upsample1D(nn.Sequential):
    
    def __init__(self, scale, num_feat):
        m=[]
        if (scale & (scale-1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv1d(num_feat, 2*num_feat, 3, 1, 1))
                m.append(PixelShuffle1D(2))
        elif scale == 3:
            m.append(nn.Conv1d, num_feat, 9*num_feat, 3, 1, 1)
            m.append(PixelShuffle1D(scale))
        else:
            raise ValueError(f'scale {scale} is not supported.' 'Supported scales: 2^n and 3')
        super(Upsample1D, self).__init__(*m)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition_1d(x, window_size, divide_half_plane=0):
    """
    Args:
        x: (B, L, C)
        window_size (int): window length

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    
    if divide_half_plane == 1:
        hsize = x.shape[1] // 2
        top_half = x[:, :hsize, :, :]
        bottom_half = x[:, hsize:, :, :]
        
        x_top = top_half.contiguous().view(-1, window_size, C)
        x_bottom = bottom_half.contiguous().view(-1, window_size, C)
        return x_top, x_bottom
    else:
        windows = x.contiguous().view(-1, window_size, C)
        return windows

def window_reverse_1d(windows, window_size, L, divide_half_plane=False):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Sequence length

    Returns:
        x: (B, L, C)
    """
    if divide_half_plane:
        # B = int(windows.shape[0] / (L / window_size))
        C = windows[0].shape[-1]
        B = int(windows[0].shape[0] * len(windows) / (L / window_size))
        x_top = windows[0]    #B*nw/2 , wsize, C
        x_bottom = windows[1] #B*nw/2 , wsize, C
        x_top = x_top.view(B, -1, window_size, C) #B, nw/2 , wsize, C
        x_bottom = x_bottom.view(B, -1, window_size, C)
        x = torch.concat((x_top, x_bottom), dim=1).contiguous().view(B, L, -1)
        
        return x
    else:
        B = int(windows.shape[0] / (L / window_size))
        x = windows.view(B, L // window_size, window_size, -1)
        x = x.contiguous().view(B, L, -1)
    return x

class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True, 
                 export_att=False, multi_rpe=1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.export_att = export_att
        self.multi_rpe = multi_rpe

        # 一维相对位置编码表
        self.rpe = rpe
        if self.rpe:
            if multi_rpe == 1:
                self.relative_position_bias_table = nn.Parameter(torch.zeros(NUM_DATE, 2 * window_size - 1, num_heads))
            elif multi_rpe == 2:
                self.relative_position_bias_table = nn.Parameter(torch.zeros(NUM_COLUMN, 2 * window_size - 1, num_heads))
            elif multi_rpe > 2:
                self.relative_position_bias_date_table = nn.Parameter(torch.zeros(NUM_DATE, 2 * window_size - 1, num_heads))
                self.relative_position_bias_col_table = nn.Parameter(torch.zeros(NUM_COLUMN, 2 * window_size - 1, num_heads))
                if multi_rpe == 4:
                    self.fuse_rpe = Mlp(in_features=2*num_heads, out_features=num_heads, act_layer=nn.GELU, drop=0.)
            else:
                self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window_size - 1, num_heads))  # [2M-1, nH]
            
            if self.multi_rpe > 2:
                nn.init.trunc_normal_(self.relative_position_bias_date_table, std=.02)
                nn.init.trunc_normal_(self.relative_position_bias_col_table, std=.02)
            else:
                nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # 生成相对位置索引
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]  # [M, M]
        relative_coords += window_size - 1  # 转换到0~2M-2范围
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, condition, mask=None):
        B_, N, C = x.shape  # N=window_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #3, B, nhead, N, c//nhead

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        # 添加相对位置偏置
        #debug_mzh
        if self.rpe:
            if self.multi_rpe == 1:#single condi: Date
                date_idx = condition
                relative_position_bias = self.relative_position_bias_table[date_idx][
                    self.relative_position_index.view(-1)].view(
                        self.window_size, self.window_size, -1)  # [M, M, nH]   
            elif self.multi_rpe == 2: #single condi: ColID
                #2M-1, nH
                tmp=self.relative_position_bias_table[condition] #[B_, 2M-1， nH]
                relative_position_bias = tmp[:, self.relative_position_index.view(-1)].view(
                        -1, self.window_size, self.window_size, self.num_heads)  # [B_, M, M, nH]
            elif self.multi_rpe > 2: #double condi: Date&&ColID
                #condition [B_, 2]
                condi_col = condition[:, 0].squeeze(-1)
                condi_date = condition[:, 1].squeeze(-1)
                tb_col = self.relative_position_bias_col_table[condi_col]   # [B_, 2M-1, nH]
                tb_date = self.relative_position_bias_date_table[condi_date] # [B_, 2M-1, nH]

                # Fuse the compact relative-distance tables before expanding
                # them to the M x M attention matrix. This is algebraically
                # equivalent to the original point-wise fusion while avoiding
                # very large intermediate activations during backward.
                if self.multi_rpe == 3:
                    fused_rpe_table = tb_col + tb_date
                elif self.multi_rpe == 4:
                    fused_rpe_table = self.fuse_rpe(torch.concat([tb_col, tb_date], dim=-1))

                relative_position_bias = fused_rpe_table[
                    :, self.relative_position_index.view(-1)
                ].view(-1, self.window_size, self.window_size, self.num_heads)
            else:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(
                        self.window_size, self.window_size, -1)  # [M, M, nH]
                
            
            if self.multi_rpe >= 2:
                relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()  # [B_, nH, M, M]
                attn = attn + relative_position_bias
            else: 
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, M, M]
                attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            mask_final = mask.unsqueeze(0).unsqueeze(0)
            mask_final = mask_final.expand(B_, self.num_heads, -1, -1).bool()
            attn = attn.masked_fill(mask_final, -1e12)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        #up_triangle mask
        # mask = torch.triu(torch.ones(self.window_size, self.window_size), diagonal=0).to(attn.device)
        #down triangle mask
        # mask = torch.tril(torch.ones(self.window_size, self.window_size), diagonal=0).to(attn.device)
        # mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1).unsqueeze(0).expand(B_, -1, -1, -1)
        # attn = attn * mask
        
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.export_att:
            #debug
            # return x, attn
            return x, relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)
        else:
            return x, torch.zeros(0).to(x.device)

class SwinTransformerBlock1D(nn.Module):
    def __init__(self, dim, input_length, num_heads, window_size=4096,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rpe=True, export_att=False, half_att=False, multi_rpe=1):
        super().__init__()
        self.dim = dim
        self.input_length = input_length
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.export_att = export_att
        self.half_att = half_att
        if self.half_att:
            triu_mask = torch.triu(torch.ones(self.window_size, self.window_size), diagonal=1)
            tril_mask = torch.tril(torch.ones(self.window_size, self.window_size), diagonal=-1)
            self.register_buffer("att_triu_mask", triu_mask)
            self.register_buffer("att_tril_mask", tril_mask)

        if input_length <= window_size:
            self.window_size = input_length
        assert 0 < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention1D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
            export_att=self.export_att, multi_rpe=multi_rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, condition):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 窗口划分/windows divide
        if self.half_att:
            xtop_windows, xbottom_windows = window_partition_1d(x, self.window_size, divide_half_plane=True)
        else:
            x_windows = window_partition_1d(x, self.window_size, divide_half_plane=False)  # [B*nW, wsize, C]
        
        # 窗口注意力/window-based attention
        # if self.export_att:
        #     attn_windows, att_weight = self.attn(x_windows, date_str) #[B*nW, wsize, C]  [B*nW, nh, wsize, wsize]
        # else:
        #     attn_windows = self.attn(x_windows, date_str)
        
        if self.half_att:
            att_top, att_wt_top = self.attn(xtop_windows, condition, mask=self.att_triu_mask)#bottom remained
            att_bottom, att_wt_bottom = self.attn(xbottom_windows, condition, mask=self.att_tril_mask)
            if not att_wt_top.numel() == 0 and (not att_wt_bottom.numel() == 0):
                att_weight = att_wt_top + att_wt_bottom
        else:
            attn_windows, att_weight = self.attn(x_windows, condition) #[B*nW, wsize, C]  [B*nW, nh, wsize, wsize]

        # 合并窗口
        if self.half_att:
            x = window_reverse_1d([att_top, att_bottom], self.window_size, L, divide_half_plane=True)    
        else:
            x = window_reverse_1d(attn_windows, self.window_size, L, divide_half_plane=False)  # [B, L, C]
            
        x = shortcut + self.drop_path(x)
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))

        if self.export_att:
            return x, att_weight
        else:
            return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, export_att=False, multi_rpe=1, half_att=False,
                 rpe=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.export_att = export_att

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock1D(dim=dim, input_length=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, export_att = export_att, half_att=half_att, multi_rpe= multi_rpe,
                                 rpe=rpe)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, condition):
        att_wt_layers = []

        for blk in self.blocks:
            if self.export_att:
                x, att_wt = blk(x, condition)
                att_wt_layers.append(torch.unsqueeze(att_wt.clone().detach(), dim=0))
            else:
                x = blk(x, condition)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.export_att:
            att_wt_layers = torch.concat((att_wt_layers), dim=0) #[nsublayer, B*nW, nh, wsize, wsize]
            return x, att_wt_layers
        else:
            return x

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        img_size: Input image size.
        patch_size: Patch size.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 seq_len=224, patch_size=4,
                 rpe=True, residual=True, export_att=False, half_att=False, multi_rpe=1):
        super(RSTB, self).__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.residual = residual
        self.export_att = export_att

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         export_att=export_att,
                                         multi_rpe=multi_rpe,
                                         half_att=half_att,
                                         rpe=rpe)

        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            seq_len=seq_len, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            seq_len=seq_len, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size, condition):
        #x[B, Ntokens, C]
        
        if self.export_att:
            x_atted, att_wt = self.residual_group(x, condition)
            if self.residual:
                return self.patch_embed(self.conv(self.patch_unembed(x_atted, x_size))) + x, att_wt
            else:
                return self.patch_embed(self.conv(self.patch_unembed(x_atted, x_size))), att_wt
        else:
            if self.residual:
                return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, condition), x_size))) + x
            else:
                return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, condition), x_size)))

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        seq_len (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, seq_len=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
 
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #[B, ch, seq_len]

        x = x.transpose(1, 2)  # [B num_patches C]
            
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        seq_len (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, seq_len=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
            
    def forward(self, x, x_size):
        B, num_batches, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, -1)  # B Ph*Pw C
        return x

class DeCTIMPE(nn.Module):
    r""" DeCTI

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
    """

    def __init__(self, seq_len=64, seq_width=4096, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=4096, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 ape=True, rpe=True, residual=True, export_att=False, half_att=False, multi_ape=1, multi_rpe=1, updown_version=2, 
                 **kwargs):
        super(DeCTIMPE, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        if (not multi_ape) and (not multi_rpe):
            assert multi_ape == multi_rpe

        self.seq_width = seq_width
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.updown_version=updown_version
        if patch_size > 1:
            if updown_version == 1:
                self.conv_first = nn.Conv1d(num_in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
                self.conv_last = nn.Conv1d(embed_dim, num_out_ch, 3, 1, 1)
                self.upsample = Upsample1D(patch_size,embed_dim)
            else:
                embed_dim *= patch_size
                self.conv_first = nn.Conv1d(num_in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
                self.conv_last = nn.Conv1d(embed_dim//patch_size, num_out_ch, 3, 1, 1)
                self.upsample = PixelShuffle1D(patch_size)
        else:
            self.conv_first = nn.Conv1d(num_in_ch, embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv1d(embed_dim, num_out_ch, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.residual = residual
        self.export_att = export_att
        self.multi_ape = multi_ape
        self.multi_rpe = multi_rpe
        
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            seq_len=seq_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.num_patches = self.patch_embed.num_patches

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            seq_len=seq_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            if self.multi_ape == 1:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(NUM_DATE, 1, self.num_patches, embed_dim))
            elif self.multi_ape == 2:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(NUM_COLUMN, 1, self.num_patches, embed_dim))
            elif self.multi_ape > 2: #add two position encoding
                assert self.multi_ape == self.multi_rpe
                self.absolute_pos_embed_date = nn.Parameter(torch.zeros(NUM_DATE, 1, self.num_patches, embed_dim))
                self.absolute_pos_embed_col = nn.Parameter(torch.zeros(NUM_COLUMN, 1, self.num_patches, embed_dim))
                if self.multi_ape == 4:
                    self.fuse_ape = Mlp(in_features=2*embed_dim, hidden_features=embed_dim, out_features=embed_dim, act_layer=nn.GELU, drop=0.)
            else:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

            if self.multi_ape > 2:
                trunc_normal_(self.absolute_pos_embed_date, std=.02)
                trunc_normal_(self.absolute_pos_embed_col, std=.02)
            else:
                trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                        input_resolution=self.num_patches,
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                        norm_layer=norm_layer,
                        downsample=None,
                        seq_len=seq_len,
                        patch_size=patch_size,
                        rpe=rpe, export_att=export_att, half_att=half_att, multi_rpe=multi_rpe, residual=residual,
                        )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv1d(embed_dim, embed_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  

    def check_image_size(self, x):
        _, _, seq_len = x.size()
        
        mod_pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        if mod_pad_len > 0:
            return F.pad(x, (0, mod_pad_len), 'reflect')
        else:
            return x

    def date_id_mapping(self, condition):
        return int(date_id_mapping(condition, num_bins=NUM_DATE))

    def forward_features(self, x, condition):
        x_size = x.shape[2]
        
        x = self.patch_embed(x)   #[B, num_patches, C]
        
        condition_att = None
        if self.ape:
            if self.multi_ape == 2 or self.multi_rpe == 2:
                condition = condition // (self.seq_width // NUM_COLUMN)
                condition = torch.clamp(condition, min=0, max=NUM_COLUMN-1) #[batch_size, idx]
                
                if self.multi_ape == 2:
                    ap_embed = self.absolute_pos_embed[condition]
                else:
                    ap_embed = self.absolute_pos_embed
                
                if self.multi_rpe == 2:
                    nwindow = x_size // self.window_size
                    condition_att = torch.repeat_interleave(condition, nwindow, dim=0)
                else:
                    condition_att = condition
                    
                x = x + torch.squeeze(ap_embed, dim=1)
            elif self.multi_ape > 2 or self.multi_rpe > 2:
                condition[:, 0] = condition[:, 0] // NUM_COLUMN
                condi_col = condition[:, 0].squeeze(-1)
                condi_date = condition[:, 1].squeeze(-1)

                if self.multi_ape ==  3:
                    ap_embed = self.absolute_pos_embed_col[condi_col] + self.absolute_pos_embed_date[condi_date]
                elif self.multi_ape == 4:
                    ap_embed = self.fuse_ape(torch.concat([self.absolute_pos_embed_col[condi_col], self.absolute_pos_embed_date[condi_date]], dim=-1)) #[1, 1, ntoken, dim]
                
                nwindow = x_size // self.window_size
                condition_att = torch.repeat_interleave(condition, nwindow, dim=0) #[nwindow*batch_size, 2]

                x = x + torch.squeeze(ap_embed, dim=1)
            elif self.multi_ape == 1:
                condition_att = condition
                x = x + self.absolute_pos_embed[condition_att]
            else:
                condition_att = condition
                x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        if self.export_att:
            att_wt_mlayers = []
            for layer in self.layers:
                x, att_sublayers = layer(x, x_size, condition_att)
                att_wt_mlayers.append(torch.unsqueeze(att_sublayers.clone().detach(), dim=0))
            att_wt_mlayers = torch.concat((att_wt_mlayers), dim=0) #[nlayer, nsublayer, B*nW, nh, wsize, wsize]
        else:
            for layer in self.layers:
                x = layer(x, x_size, condition_att)

        x = self.norm(x)  #[B, L, C]
        x = self.patch_unembed(x, x_size)

        if self.export_att:
            return x, att_wt_mlayers
        else:
            return x

    def forward(self, x, condition):

        x = x.permute(1,2,0)    # x: [Input_length, Batch, Channel]-> [Batch, C, Input_length]
        
        seq_len = x.shape[2]
        x = self.check_image_size(x)

        x_first = self.conv_first(x)
        
        if self.export_att:
            x_feat, att_wt_mlayers = self.forward_features(x_first, condition)
        else:
            x_feat = self.forward_features(x_first, condition)
            
        if self.residual:
            res = self.conv_after_body(x_feat) + x_first
            if self.patch_size > 1:
                res = self.upsample(res)
            x = x + self.conv_last(res)
        else:
            res = self.conv_after_body(x_feat)
            if self.patch_size > 1:
                res = self.upsample(res)
            x = self.conv_last(res)

        x = x[:, :, :seq_len].permute(2,0,1)    # x: [Input_length, Batch, Channel]

        if self.export_att:
            return x, att_wt_mlayers
        else:
            return x, torch.zeros(0).to(x.device)
