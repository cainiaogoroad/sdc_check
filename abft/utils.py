# abft/utils.py

"""
ABFT工具函数，用于计算和验证校验和
"""

import torch
import numpy as np
from typing import Union, Dict, Any

TensorOrArray = Union[torch.Tensor, np.ndarray]

def checksum(tensor: TensorOrArray) -> Dict[str, Any]:
    """
    计算张量的校验和。
    
    支持PyTorch张量和NumPy数组。
    
    Args:
        tensor: 输入的PyTorch张量或NumPy数组。
        
    Returns:
        一个包含多种校验和的字典，例如：
        - 'sum': 张量所有元素的总和。
        - 'mean': 张量所有元素的平均值。
        - 'std': 张量所有元素的标准差。
        - 'row_sum': (对于2D或更高维度) 张量在最后一个维度的和。
        - 'col_sum': (对于2D或更高维度) 张量在第一个维度的和。
    """
    if isinstance(tensor, torch.Tensor):
        # PyTorch张量
        if tensor.dim() == 1:
            # 一维张量
            return {
                'sum': tensor.sum().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item() if tensor.numel() > 1 else 0.0
            }
        else:
            # 二维或更高维张量
            checksums = {
                'row_sum': torch.sum(tensor, dim=-1).detach().clone(),
                'sum': tensor.sum().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item()
            }
            if tensor.dim() > 1:
                checksums['col_sum'] = torch.sum(tensor, dim=0).detach().clone()
            return checksums
    else:
        # NumPy数组
        if tensor.ndim == 1:
            # 一维数组
            return {
                'sum': np.sum(tensor).item(),
                'mean': np.mean(tensor).item(),
                'std': np.std(tensor).item() if tensor.size > 1 else 0.0
            }
        else:
            # 二维或更高维数组
            checksums = {
                'row_sum': np.sum(tensor, axis=-1).copy(),
                'sum': np.sum(tensor).item(),
                'mean': np.mean(tensor).item(),
                'std': np.std(tensor).item()
            }
            if tensor.ndim > 1:
                checksums['col_sum'] = np.sum(tensor, axis=0).copy()
            return checksums

def verify_checksum(tensor: TensorOrArray, original_checksum: Dict[str, Any], tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    验证张量校验和是否与原始校验和匹配。
    
    Args:
        tensor: 要验证的张量。
        original_checksum: 原始校验和字典。
        tolerance: 容差阈值，用于浮点数比较。
        
    Returns:
        一个包含验证结果的字典，详细说明了哪些校验和不匹配。
    """
    # 计算当前校验和
    current_checksum = checksum(tensor)
    
    # 验证基本指标
    sum_diff = abs(current_checksum['sum'] - original_checksum['sum'])
    sum_corrupted = sum_diff > tolerance
    
    mean_diff = abs(current_checksum['mean'] - original_checksum['mean'])
    mean_corrupted = mean_diff > tolerance
    
    std_diff = abs(current_checksum['std'] - original_checksum['std'])
    std_corrupted = std_diff > tolerance
    
    # 检查行和列校验和
    row_corrupted, col_corrupted = False, False
    corrupted_rows, corrupted_cols = [], []
    
    if 'row_sum' in original_checksum and 'row_sum' in current_checksum:
        row_diff = np.abs(current_checksum['row_sum'] - original_checksum['row_sum'])
        row_corrupted = np.any(row_diff > tolerance).item()
        if row_corrupted:
            corrupted_rows = np.where(row_diff > tolerance)[0].tolist()

    if 'col_sum' in original_checksum and 'col_sum' in current_checksum:
        col_diff = np.abs(current_checksum['col_sum'] - original_checksum['col_sum'])
        col_corrupted = np.any(col_diff > tolerance).item()
        if col_corrupted:
            corrupted_cols = np.where(col_diff > tolerance)[0].tolist()

    is_corrupted = sum_corrupted or mean_corrupted or std_corrupted or row_corrupted or col_corrupted
    
    return {
        'is_corrupted': is_corrupted,
        'sum_corrupted': sum_corrupted,
        'mean_corrupted': mean_corrupted,
        'std_corrupted': std_corrupted,
        'row_corrupted': row_corrupted,
        'col_corrupted': col_corrupted,
        'corrupted_rows': corrupted_rows,
        'corrupted_cols': corrupted_cols,
        'diff_sum': sum_diff,
        'diff_mean': mean_diff,
        'diff_std': std_diff
    }
