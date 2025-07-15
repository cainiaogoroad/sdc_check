"""
实现简单的多层感知机(MLP)，并支持ABFT检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    简单的多层感知机实现
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # 用于ABFT的状态
        self.last_fc1_output = None
        self.last_fc2_input = None
        self.last_checksums = None
    
    def forward(self, x, enable_abft=False):
        """
        Args:
            x: 输入张量
            enable_abft: 是否启用ABFT检测
            
        Returns:
            输出张量
        """
        # 第一个线性层
        x = self.fc1(x)
        
        # 如果启用ABFT，保存第一层输出用于检测
        if enable_abft:
            self.last_fc1_output = x.detach().clone()
            # 计算行和和列和作为校验和
            self.last_checksums = {
                'fc1_output_row_sum': torch.sum(x, dim=-1),
                'fc1_output_col_sum': torch.sum(x, dim=0) if len(x.shape) > 1 else x
            }
        
        # 激活函数
        x = self.act(x)
        
        # 如果启用ABFT，保存激活后的输出
        if enable_abft:
            self.last_fc2_input = x.detach().clone()
            self.last_checksums.update({
                'fc2_input_row_sum': torch.sum(x, dim=-1),
                'fc2_input_col_sum': torch.sum(x, dim=0) if len(x.shape) > 1 else x
            })
        
        # 第二个线性层和dropout
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    
    def verify_integrity(self, layer='fc1', current_tensor=None):
        """
        验证指定层输出的完整性，检测是否有SDC
        
        Args:
            layer: 要验证的层，可以是'fc1'或'fc2_input'
            current_tensor: 当前的张量值，如果为None则使用保存的值
            
        Returns:
            检测结果字典，包含是否检测到SDC及相关信息
        """
        if self.last_checksums is None:
            return {'is_corrupted': False, 'reason': 'No saved checksum found'}
        
        if layer == 'fc1':
            saved_tensor = self.last_fc1_output
            row_checksum_key = 'fc1_output_row_sum'
            col_checksum_key = 'fc1_output_col_sum'
        elif layer == 'fc2_input':
            saved_tensor = self.last_fc2_input
            row_checksum_key = 'fc2_input_row_sum'
            col_checksum_key = 'fc2_input_col_sum'
        else:
            return {'is_corrupted': False, 'reason': f'Unknown layer: {layer}'}
        
        tensor = current_tensor if current_tensor is not None else saved_tensor
        if tensor is None:
            return {'is_corrupted': False, 'reason': f'No tensor saved for layer: {layer}'}
        
        # 计算当前张量的校验和
        current_row_sum = torch.sum(tensor, dim=-1)
        current_col_sum = torch.sum(tensor, dim=0) if len(tensor.shape) > 1 else tensor
        
        # 验证行和
        row_diff = torch.abs(current_row_sum - self.last_checksums[row_checksum_key])
        row_corrupted = torch.any(row_diff > 1e-5)
        
        # 验证列和
        col_diff = torch.abs(current_col_sum - self.last_checksums[col_checksum_key])
        col_corrupted = torch.any(col_diff > 1e-5)
        
        is_corrupted = row_corrupted or col_corrupted
        
        # 如果检测到损坏，找出损坏位置
        corrupted_indices = None
        if is_corrupted:
            # 找出损坏的行和列
            corrupted_rows = torch.where(row_diff > 1e-5)[0].cpu().numpy().tolist() if len(tensor.shape) > 1 else []
            corrupted_cols = torch.where(col_diff > 1e-5)[0].cpu().numpy().tolist()
            corrupted_indices = {
                'rows': corrupted_rows,
                'cols': corrupted_cols
            }
        
        return {
            'is_corrupted': is_corrupted,
            'row_corrupted': row_corrupted.item() if isinstance(row_corrupted, torch.Tensor) else row_corrupted,
            'col_corrupted': col_corrupted.item() if isinstance(col_corrupted, torch.Tensor) else col_corrupted,
            'corrupted_indices': corrupted_indices
        } 