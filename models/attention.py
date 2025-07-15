"""
实现简单的自注意力机制(Self-Attention)，并支持ABFT检测
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    简单的自注意力机制实现
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 用于ABFT的状态
        self.last_attn_weights = None
        self.last_checksum = None
    
    def forward(self, x, enable_abft=False):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            enable_abft: 是否启用ABFT检测
            
        Returns:
            输出张量，形状与输入相同
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q,k,v: [B, num_heads, N, head_dim]
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 如果启用ABFT，保存注意力权重用于检测
        if enable_abft:
            self.last_attn_weights = attn.detach().clone()
            # 计算行和列的校验和作为ABFT检测的基础
            self.last_checksum = {
                'row_sum': torch.sum(attn, dim=-1),  # 行和
                'col_sum': torch.sum(attn, dim=-2)   # 列和
            }
        
        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def verify_attn_integrity(self, current_attn=None):
        """
        验证注意力权重的完整性，检测是否有SDC
        
        Args:
            current_attn: 当前的注意力权重，如果为None则使用last_attn_weights
            
        Returns:
            检测结果字典，包含是否检测到SDC及相关信息
        """
        if self.last_checksum is None:
            return {'is_corrupted': False, 'reason': 'No saved checksum found'}
        
        attn = current_attn if current_attn is not None else self.last_attn_weights
        if attn is None:
            return {'is_corrupted': False, 'reason': 'No attention weights saved'}
        
        # 计算当前注意力权重的校验和
        current_row_sum = torch.sum(attn, dim=-1)
        current_col_sum = torch.sum(attn, dim=-2)
        
        # 验证行和
        row_diff = torch.abs(current_row_sum - self.last_checksum['row_sum'])
        row_corrupted = torch.any(row_diff > 1e-5)
        
        # 验证列和
        col_diff = torch.abs(current_col_sum - self.last_checksum['col_sum'])
        col_corrupted = torch.any(col_diff > 1e-5)
        
        is_corrupted = row_corrupted or col_corrupted
        
        # 如果检测到损坏，找出损坏位置
        corrupted_indices = None
        if is_corrupted:
            # 找出损坏的行和列
            corrupted_rows = torch.where(torch.any(row_diff > 1e-5, dim=-1))[0]
            corrupted_cols = torch.where(torch.any(col_diff > 1e-5, dim=-1))[0]
            corrupted_indices = {
                'rows': corrupted_rows.cpu().numpy().tolist(),
                'cols': corrupted_cols.cpu().numpy().tolist()
            }
        
        return {
            'is_corrupted': is_corrupted,
            'row_corrupted': row_corrupted.item() if isinstance(row_corrupted, torch.Tensor) else row_corrupted,
            'col_corrupted': col_corrupted.item() if isinstance(col_corrupted, torch.Tensor) else col_corrupted,
            'corrupted_indices': corrupted_indices
        } 