"""
实现简单的自注意力机制(Self-Attention)，并支持ABFT检测
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class Attention(nn.Module):
    """
    简单的自注意力机制实现
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 用于ABFT的状态
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.last_checksum: Optional[Dict[str, torch.Tensor]] = None
    
    def forward(self, x: torch.Tensor, enable_abft: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            enable_abft: 是否启用ABFT检测
            
        Returns:
            输出张量，形状与输入相同
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if enable_abft:
            self.last_attn_weights = attn.detach().clone()
            self.last_checksum = {
                'row_sum': torch.sum(attn, dim=-1),
                'col_sum': torch.sum(attn, dim=-2)
            }
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def verify_attn_integrity(self, current_attn: Optional[torch.Tensor] = None, tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        验证注意力权重的完整性，检测是否有SDC。
        
        Args:
            current_attn: 当前的注意力权重。如果为None，则使用前向传播时保存的权重。
            tolerance: 容差阈值，用于浮点数比较。
            
        Returns:
            一个包含检测结果的字典。
        """
        if self.last_checksum is None:
            return {'is_corrupted': False, 'reason': 'No saved checksum found'}
        
        attn = current_attn if current_attn is not None else self.last_attn_weights
        if attn is None:
            return {'is_corrupted': False, 'reason': 'No attention weights saved'}
        
        current_row_sum = torch.sum(attn, dim=-1)
        row_diff = torch.abs(current_row_sum - self.last_checksum['row_sum'])
        row_corrupted = torch.any(row_diff > tolerance)
        
        current_col_sum = torch.sum(attn, dim=-2)
        col_diff = torch.abs(current_col_sum - self.last_checksum['col_sum'])
        col_corrupted = torch.any(col_diff > tolerance)
        
        is_corrupted = row_corrupted or col_corrupted
        
        corrupted_indices = None
        if is_corrupted:
            corrupted_rows = torch.where(torch.any(row_diff > tolerance, dim=-1))[0].cpu().numpy().tolist()
            corrupted_cols = torch.where(torch.any(col_diff > tolerance, dim=-1))[0].cpu().numpy().tolist()
            corrupted_indices = {'rows': corrupted_rows, 'cols': corrupted_cols}
        
        return {
            'is_corrupted': is_corrupted.item(),
            'row_corrupted': row_corrupted.item(),
            'col_corrupted': col_corrupted.item(),
            'corrupted_indices': corrupted_indices
        }
