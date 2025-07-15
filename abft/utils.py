"""
ABFT工具函数，用于计算和验证校验和
"""

import torch
import numpy as np

def checksum(tensor):
    """
    计算张量的校验和
    
    支持PyTorch张量和NumPy数组
    
    Args:
        tensor: 输入张量或数组
        
    Returns:
        校验和字典，包含行和、列和、总和等
    """
    if isinstance(tensor, torch.Tensor):
        # PyTorch张量
        if tensor.dim() == 1:
            # 一维张量只有总和
            return {
                'sum': tensor.sum().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item() if tensor.numel() > 1 else 0.0
            }
        else:
            # 二维或更高维张量
            checksums = {
                'row_sum': torch.sum(tensor, dim=-1).detach().clone(),  # 最后一维求和
                'sum': tensor.sum().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item()
            }
            
            # 如果是二维以上，还可以计算其他维度的和
            if tensor.dim() > 1:
                checksums['col_sum'] = torch.sum(tensor, dim=0).detach().clone()  # 第一维求和
            
            return checksums
    else:
        # NumPy数组
        if tensor.ndim == 1:
            # 一维数组只有总和
            return {
                'sum': np.sum(tensor).item(),
                'mean': np.mean(tensor).item(),
                'std': np.std(tensor).item() if tensor.size > 1 else 0.0
            }
        else:
            # 二维或更高维数组
            checksums = {
                'row_sum': np.sum(tensor, axis=-1).copy(),  # 最后一维求和
                'sum': np.sum(tensor).item(),
                'mean': np.mean(tensor).item(),
                'std': np.std(tensor).item()
            }
            
            # 如果是二维以上，还可以计算其他维度的和
            if tensor.ndim > 1:
                checksums['col_sum'] = np.sum(tensor, axis=0).copy()  # 第一维求和
            
            return checksums

def verify_checksum(tensor, original_checksum, tolerance=1e-5):
    """
    验证张量校验和是否与原始校验和匹配
    
    Args:
        tensor: 要验证的张量
        original_checksum: 原始校验和
        tolerance: 容差阈值
        
    Returns:
        验证结果字典，包含是否有损坏及相关信息
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
    
    # 检查行和列校验和（如果存在）
    row_corrupted = False
    col_corrupted = False
    corrupted_rows = []
    corrupted_cols = []
    
    if 'row_sum' in original_checksum and 'row_sum' in current_checksum:
        if isinstance(original_checksum['row_sum'], torch.Tensor):
            row_diff = torch.abs(current_checksum['row_sum'] - original_checksum['row_sum'])
            row_corrupted = torch.any(row_diff > tolerance).item()
            if row_corrupted:
                corrupted_rows = torch.where(row_diff > tolerance)[0].cpu().numpy().tolist()
        else:
            row_diff = np.abs(current_checksum['row_sum'] - original_checksum['row_sum'])
            row_corrupted = np.any(row_diff > tolerance).item()
            if row_corrupted:
                corrupted_rows = np.where(row_diff > tolerance)[0].tolist()
    
    if 'col_sum' in original_checksum and 'col_sum' in current_checksum:
        if isinstance(original_checksum['col_sum'], torch.Tensor):
            col_diff = torch.abs(current_checksum['col_sum'] - original_checksum['col_sum'])
            col_corrupted = torch.any(col_diff > tolerance).item()
            if col_corrupted:
                corrupted_cols = torch.where(col_diff > tolerance)[0].cpu().numpy().tolist()
        else:
            col_diff = np.abs(current_checksum['col_sum'] - original_checksum['col_sum'])
            col_corrupted = np.any(col_diff > tolerance).item()
            if col_corrupted:
                corrupted_cols = np.where(col_diff > tolerance)[0].tolist()
    
    # 综合判断是否有损坏
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