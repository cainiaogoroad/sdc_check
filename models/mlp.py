"""
实现简单的多层感知机(MLP)，并支持ABFT检测
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type

class MLP(nn.Module):
    """
    简单的多层感知机实现
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Type[nn.Module] = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        self.last_fc1_output: Optional[torch.Tensor] = None
        self.last_fc2_input: Optional[torch.Tensor] = None
        self.last_checksums: Optional[Dict[str, Any]] = None
    
    def forward(self, x: torch.Tensor, enable_abft: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入张量
            enable_abft: 是否启用ABFT检测
            
        Returns:
            输出张量
        """
        x = self.fc1(x)
        if enable_abft:
            self.last_fc1_output = x.detach().clone()
            self.last_checksums = {
                'fc1_output_row_sum': torch.sum(x, dim=-1),
                'fc1_output_col_sum': torch.sum(x, dim=0) if x.dim() > 1 else x.clone()
            }
        
        x = self.act(x)
        if enable_abft:
            self.last_fc2_input = x.detach().clone()
            self.last_checksums.update({
                'fc2_input_row_sum': torch.sum(x, dim=-1),
                'fc2_input_col_sum': torch.sum(x, dim=0) if x.dim() > 1 else x.clone()
            })
        
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    
    def verify_integrity(self, layer: str = 'fc1', current_tensor: Optional[torch.Tensor] = None, tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        验证指定层输出的完整性，检测是否有SDC。
        
        Args:
            layer: 要验证的层，可以是 'fc1' 或 'fc2_input'。
            current_tensor: 当前的张量值。如果为None，则使用保存的值。
            tolerance: 容差阈值，用于浮点数比较。
            
        Returns:
            一个包含检测结果的字典。
        """
        if self.last_checksums is None:
            return {'is_corrupted': False, 'reason': 'No saved checksum found'}
        
        if layer == 'fc1':
            saved_tensor = self.last_fc1_output
            row_key, col_key = 'fc1_output_row_sum', 'fc1_output_col_sum'
        elif layer == 'fc2_input':
            saved_tensor = self.last_fc2_input
            row_key, col_key = 'fc2_input_row_sum', 'fc2_input_col_sum'
        else:
            return {'is_corrupted': False, 'reason': f'Unknown layer: {layer}'}
        
        tensor_to_check = current_tensor if current_tensor is not None else saved_tensor
        if tensor_to_check is None:
            return {'is_corrupted': False, 'reason': f'No tensor saved for layer: {layer}'}
        
        current_row_sum = torch.sum(tensor_to_check, dim=-1)
        row_diff = torch.abs(current_row_sum - self.last_checksums[row_key])
        row_corrupted = torch.any(row_diff > tolerance)
        
        current_col_sum = torch.sum(tensor_to_check, dim=0) if tensor_to_check.dim() > 1 else tensor_to_check
        col_diff = torch.abs(current_col_sum - self.last_checksums[col_key])
        col_corrupted = torch.any(col_diff > tolerance)
        
        is_corrupted = row_corrupted or col_corrupted
        
        corrupted_indices = None
        if is_corrupted:
            corrupted_rows = torch.where(row_diff > tolerance)[0].cpu().numpy().tolist() if tensor_to_check.dim() > 1 else []
            corrupted_cols = torch.where(col_diff > tolerance)[0].cpu().numpy().tolist()
            corrupted_indices = {'rows': corrupted_rows, 'cols': corrupted_cols}
        
        return {
            'is_corrupted': is_corrupted.item(),
            'row_corrupted': row_corrupted.item(),
            'col_corrupted': col_corrupted.item(),
            'corrupted_indices': corrupted_indices
        }
