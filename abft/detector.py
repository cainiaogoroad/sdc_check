"""
SDC检测器实现，用于检测模型计算过程中的静默数据损坏
"""

import torch
import numpy as np
from .utils import checksum, verify_checksum

class SDCDetector:
    """
    静默数据损坏(SDC)检测器
    
    通过算法基础容错(ABFT)技术检测模型计算过程中的数据损坏
    """
    
    def __init__(self, tolerance=1e-5):
        """
        初始化SDC检测器
        
        Args:
            tolerance: 检测阈值，差异小于此值时不认为是损坏
        """
        self.tolerance = tolerance
        self.checksums = {}
        self.original_tensors = {}
    
    def register_tensor(self, name, tensor):
        """
        注册需要保护的张量
        
        Args:
            name: 张量名称
            tensor: 张量数据
        """
        # 复制一份张量以防止被修改
        tensor_copy = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else np.copy(tensor)
        self.original_tensors[name] = tensor_copy
        
        # 计算校验和
        self.checksums[name] = checksum(tensor_copy)
        
        return self.checksums[name]
    
    def verify_tensor(self, name, tensor=None):
        """
        验证张量是否完整（没有SDC）
        
        Args:
            name: 张量名称
            tensor: 要验证的张量，如果为None则使用之前注册的原始张量
            
        Returns:
            验证结果字典，包含是否有损坏及相关信息
        """
        if name not in self.checksums:
            return {'is_corrupted': False, 'reason': f'Tensor {name} not registered'}
        
        current_tensor = tensor if tensor is not None else self.original_tensors[name]
        
        # 验证校验和
        result = verify_checksum(current_tensor, self.checksums[name], self.tolerance)
        
        return result
    
    def inject_fault(self, name, indices=None, magnitude=1.0):
        """
        向指定张量注入错误，用于测试ABFT检测功能
        
        Args:
            name: 张量名称
            indices: 要注入错误的位置，如果为None则随机选择
            magnitude: 错误大小
            
        Returns:
            注入错误后的张量
        """
        if name not in self.original_tensors:
            raise ValueError(f'Tensor {name} not registered')
        
        tensor = self.original_tensors[name].clone() if isinstance(self.original_tensors[name], torch.Tensor) else np.copy(self.original_tensors[name])
        
        # 根据张量维度确定注入方式
        if isinstance(tensor, torch.Tensor):
            if indices is None:
                # 随机选择位置
                if tensor.dim() == 1:
                    idx = torch.randint(0, tensor.shape[0], (1,))
                    tensor[idx] += magnitude
                elif tensor.dim() == 2:
                    i = torch.randint(0, tensor.shape[0], (1,))
                    j = torch.randint(0, tensor.shape[1], (1,))
                    tensor[i, j] += magnitude
                else:
                    # 对于高维张量，展平后注入
                    flat_tensor = tensor.flatten()
                    idx = torch.randint(0, flat_tensor.shape[0], (1,))
                    flat_tensor[idx] += magnitude
                    tensor = flat_tensor.reshape(tensor.shape)
            else:
                # 使用指定位置
                if len(indices) == 1:
                    tensor[indices[0]] += magnitude
                elif len(indices) == 2:
                    tensor[indices[0], indices[1]] += magnitude
                else:
                    raise ValueError('索引维度必须与张量维度匹配')
        else:
            # NumPy数组
            if indices is None:
                # 随机选择位置
                if tensor.ndim == 1:
                    idx = np.random.randint(0, tensor.shape[0])
                    tensor[idx] += magnitude
                elif tensor.ndim == 2:
                    i = np.random.randint(0, tensor.shape[0])
                    j = np.random.randint(0, tensor.shape[1])
                    tensor[i, j] += magnitude
                else:
                    # 对于高维张量，展平后注入
                    flat_tensor = tensor.flatten()
                    idx = np.random.randint(0, flat_tensor.shape[0])
                    flat_tensor[idx] += magnitude
                    tensor = flat_tensor.reshape(tensor.shape)
            else:
                # 使用指定位置
                if len(indices) == 1:
                    tensor[indices[0]] += magnitude
                elif len(indices) == 2:
                    tensor[indices[0], indices[1]] += magnitude
                else:
                    raise ValueError('索引维度必须与张量维度匹配')
        
        return tensor 