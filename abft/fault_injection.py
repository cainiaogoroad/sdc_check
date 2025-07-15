"""
错误注入模块，用于在张量中注入各种类型的错误以测试ABFT检测功能
"""

import torch
import numpy as np
from enum import Enum
import random
from typing import Union, Tuple, List, Dict, Optional, Any, Callable

class FaultType(Enum):
    """错误类型枚举"""
    BIT_FLIP = "bit_flip"           # 位翻转错误
    RANDOM_VALUE = "random_value"   # 随机值错误
    GAUSSIAN_NOISE = "gaussian_noise"  # 高斯噪声
    ZERO_VALUE = "zero_value"       # 值置零
    CONSTANT = "constant"           # 常量替换
    SCALING = "scaling"             # 缩放错误
    PERMUTATION = "permutation"     # 元素置换

class FaultLocation(Enum):
    """错误位置策略枚举"""
    RANDOM = "random"               # 随机位置
    SPECIFIED = "specified"         # 指定位置
    PATTERN = "pattern"             # 按模式选择
    BLOCK = "block"                 # 块区域
    STRUCTURED = "structured"       # 结构化选择
    
class FaultPattern(Enum):
    """错误模式枚举"""
    SINGLE = "single"               # 单点错误
    MULTIPLE = "multiple"           # 多点错误
    ROW = "row"                     # 整行错误
    COLUMN = "column"               # 整列错误
    DIAGONAL = "diagonal"           # 对角线错误
    CROSS = "cross"                 # 十字形错误
    RANDOM_BLOCK = "random_block"   # 随机块错误

class FaultTiming(Enum):
    """错误注入时机枚举"""
    IMMEDIATE = "immediate"         # 立即注入
    DELAYED = "delayed"             # 延迟注入
    INTERMITTENT = "intermittent"   # 间歇性错误
    PERMANENT = "permanent"         # 永久性错误

class FaultInjector:
    """
    错误注入器基类，提供通用的错误注入功能
    """
    
    def __init__(self, fault_type: FaultType = FaultType.BIT_FLIP, 
                 fault_location: FaultLocation = FaultLocation.RANDOM,
                 fault_pattern: FaultPattern = FaultPattern.SINGLE,
                 fault_timing: FaultTiming = FaultTiming.IMMEDIATE,
                 probability: float = 1.0,
                 seed: Optional[int] = None):
        """
        初始化错误注入器
        
        Args:
            fault_type: 错误类型
            fault_location: 错误位置策略
            fault_pattern: 错误模式
            fault_timing: 错误注入时机
            probability: 注入错误的概率
            seed: 随机数种子
        """
        self.fault_type = fault_type
        self.fault_location = fault_location
        self.fault_pattern = fault_pattern
        self.fault_timing = fault_timing
        self.probability = probability
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            
        # 注入历史记录
        self.injection_history = []
    
    def inject(self, tensor: Union[torch.Tensor, np.ndarray], 
               indices: Optional[Union[Tuple, List]] = None, 
               **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        在张量中注入错误
        
        Args:
            tensor: 要注入错误的张量
            indices: 要注入错误的位置索引，如果为None则根据策略选择位置
            **kwargs: 额外参数
            
        Returns:
            注入错误后的张量
        """
        # 检查是否应该注入错误（基于概率）
        if random.random() > self.probability:
            return tensor
            
        # 复制张量以防止修改原始数据
        if isinstance(tensor, torch.Tensor):
            corrupted_tensor = tensor.clone()
        else:  # numpy array
            corrupted_tensor = np.copy(tensor)
            
        # 获取错误位置
        target_indices = self._get_fault_indices(corrupted_tensor, indices, **kwargs)
        
        # 根据错误类型注入错误
        corrupted_tensor = self._inject_fault(corrupted_tensor, target_indices, **kwargs)
        
        # 记录注入历史
        self.injection_history.append({
            'fault_type': self.fault_type,
            'fault_pattern': self.fault_pattern,
            'indices': target_indices,
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
        })
        
        return corrupted_tensor
    
    def _get_fault_indices(self, tensor: Union[torch.Tensor, np.ndarray], 
                          indices: Optional[Union[Tuple, List]], 
                          **kwargs) -> List:
        """
        获取要注入错误的索引位置
        
        Args:
            tensor: 张量
            indices: 用户指定的索引
            **kwargs: 额外参数
            
        Returns:
            索引列表
        """
        if indices is not None:
            # 使用用户指定的索引
            return indices if isinstance(indices, list) else [indices]
            
        shape = tensor.shape
        ndim = len(shape)
        
        if self.fault_location == FaultLocation.RANDOM:
            # 随机选择位置
            if self.fault_pattern == FaultPattern.SINGLE:
                # 单点错误
                idx = []
                for dim_size in shape:
                    idx.append(random.randint(0, dim_size - 1))
                return [tuple(idx)]
                
            elif self.fault_pattern == FaultPattern.MULTIPLE:
                # 多点错误
                num_faults = kwargs.get('num_faults', 3)
                indices_list = []
                for _ in range(num_faults):
                    idx = []
                    for dim_size in shape:
                        idx.append(random.randint(0, dim_size - 1))
                    indices_list.append(tuple(idx))
                return indices_list
                
            elif self.fault_pattern == FaultPattern.ROW:
                # 整行错误
                if ndim < 2:
                    return [(i,) for i in range(shape[0])]
                
                row_idx = random.randint(0, shape[0] - 1)
                return [(row_idx, j) for j in range(shape[1])]
                
            elif self.fault_pattern == FaultPattern.COLUMN:
                # 整列错误
                if ndim < 2:
                    return [(i,) for i in range(shape[0])]
                
                col_idx = random.randint(0, shape[1] - 1)
                return [(i, col_idx) for i in range(shape[0])]
                
            elif self.fault_pattern == FaultPattern.DIAGONAL:
                # 对角线错误
                if ndim < 2:
                    return [(i,) for i in range(shape[0])]
                
                min_dim = min(shape[0], shape[1])
                return [(i, i) for i in range(min_dim)]
                
            elif self.fault_pattern == FaultPattern.CROSS:
                # 十字形错误
                if ndim < 2:
                    return [(i,) for i in range(shape[0])]
                
                center_row = shape[0] // 2
                center_col = shape[1] // 2
                
                indices_list = []
                # 水平线
                for j in range(shape[1]):
                    indices_list.append((center_row, j))
                # 垂直线
                for i in range(shape[0]):
                    if i != center_row:  # 避免重复
                        indices_list.append((i, center_col))
                        
                return indices_list
                
            elif self.fault_pattern == FaultPattern.RANDOM_BLOCK:
                # 随机块错误
                if ndim < 2:
                    return [(i,) for i in range(shape[0])]
                
                block_size = kwargs.get('block_size', (3, 3))
                if isinstance(block_size, int):
                    block_size = (block_size, block_size)
                
                # 确保块大小不超过张量尺寸
                block_h = min(block_size[0], shape[0])
                block_w = min(block_size[1], shape[1])
                
                # 随机选择块的左上角位置
                start_h = random.randint(0, shape[0] - block_h)
                start_w = random.randint(0, shape[1] - block_w)
                
                indices_list = []
                for i in range(start_h, start_h + block_h):
                    for j in range(start_w, start_w + block_w):
                        indices_list.append((i, j))
                
                return indices_list
                
        elif self.fault_location == FaultLocation.PATTERN:
            # 按特定模式选择位置
            pattern_type = kwargs.get('pattern_type', 'checkerboard')
            
            if pattern_type == 'checkerboard':
                # 棋盘格模式
                if ndim < 2:
                    return [(i,) for i in range(0, shape[0], 2)]
                
                indices_list = []
                for i in range(shape[0]):
                    for j in range((i % 2), shape[1], 2):
                        indices_list.append((i, j))
                return indices_list
                
            elif pattern_type == 'border':
                # 边界模式
                if ndim < 2:
                    return [(0,), (shape[0]-1,)]
                
                indices_list = []
                # 上边界
                for j in range(shape[1]):
                    indices_list.append((0, j))
                # 下边界
                for j in range(shape[1]):
                    indices_list.append((shape[0]-1, j))
                # 左边界
                for i in range(1, shape[0]-1):
                    indices_list.append((i, 0))
                # 右边界
                for i in range(1, shape[0]-1):
                    indices_list.append((i, shape[1]-1))
                
                return indices_list
                
        elif self.fault_location == FaultLocation.STRUCTURED:
            # 结构化选择
            structure_type = kwargs.get('structure_type', 'gradient')
            
            if structure_type == 'gradient':
                # 梯度影响更大的位置
                if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                    # 找出梯度绝对值最大的n个位置
                    n = kwargs.get('n', 5)
                    grad_abs = torch.abs(tensor.grad).view(-1)
                    _, indices = torch.topk(grad_abs, min(n, grad_abs.numel()))
                    
                    # 转换为多维索引
                    indices_list = []
                    for idx in indices:
                        indices_list.append(np.unravel_index(idx.item(), shape))
                    
                    return indices_list
            
            # 默认返回随机单点
            idx = []
            for dim_size in shape:
                idx.append(random.randint(0, dim_size - 1))
            return [tuple(idx)]
                
        # 默认返回随机单点
        idx = []
        for dim_size in shape:
            idx.append(random.randint(0, dim_size - 1))
        return [tuple(idx)]
    
    def _inject_fault(self, tensor: Union[torch.Tensor, np.ndarray], 
                     indices: List, 
                     **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        在指定位置注入错误
        
        Args:
            tensor: 张量
            indices: 位置索引列表
            **kwargs: 额外参数
            
        Returns:
            注入错误后的张量
        """
        if self.fault_type == FaultType.BIT_FLIP:
            return self._inject_bit_flip(tensor, indices, **kwargs)
        elif self.fault_type == FaultType.RANDOM_VALUE:
            return self._inject_random_value(tensor, indices, **kwargs)
        elif self.fault_type == FaultType.GAUSSIAN_NOISE:
            return self._inject_gaussian_noise(tensor, indices, **kwargs)
        elif self.fault_type == FaultType.ZERO_VALUE:
            return self._inject_zero_value(tensor, indices)
        elif self.fault_type == FaultType.CONSTANT:
            return self._inject_constant(tensor, indices, **kwargs)
        elif self.fault_type == FaultType.SCALING:
            return self._inject_scaling(tensor, indices, **kwargs)
        elif self.fault_type == FaultType.PERMUTATION:
            return self._inject_permutation(tensor, indices, **kwargs)
        else:
            # 默认使用随机值
            return self._inject_random_value(tensor, indices, **kwargs)
    
    def _inject_bit_flip(self, tensor: Union[torch.Tensor, np.ndarray], 
                        indices: List, 
                        **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入位翻转错误"""
        bit_position = kwargs.get('bit_position', None)
        num_bits = kwargs.get('num_bits', 1)
        
        for idx in indices:
            original_value = tensor[idx].item() if isinstance(tensor[idx], (torch.Tensor, np.ndarray)) else tensor[idx]
            
            # 获取浮点数的二进制表示
            if isinstance(original_value, float):
                # 对于PyTorch，使用IEEE 754格式
                binary = float_to_binary(original_value)
                
                if bit_position is None:
                    # 随机选择位置（避开符号位）
                    positions = random.sample(range(1, len(binary)), num_bits)
                else:
                    positions = [bit_position] if isinstance(bit_position, int) else bit_position
                
                # 翻转选定位
                for pos in positions:
                    binary = binary[:pos] + ('1' if binary[pos] == '0' else '0') + binary[pos+1:]
                
                # 转回浮点数
                corrupted_value = binary_to_float(binary)
            else:
                # 对于整数类型
                bit_width = kwargs.get('bit_width', 32)  # 默认32位
                
                if bit_position is None:
                    # 随机选择位置
                    positions = random.sample(range(bit_width), num_bits)
                else:
                    positions = [bit_position] if isinstance(bit_position, int) else bit_position
                
                # 翻转选定位
                corrupted_value = original_value
                for pos in positions:
                    corrupted_value ^= (1 << pos)
            
            # 应用错误值
            tensor[idx] = corrupted_value
            
        return tensor
    
    def _inject_random_value(self, tensor: Union[torch.Tensor, np.ndarray], 
                            indices: List, 
                            **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入随机值错误"""
        min_val = kwargs.get('min_val', -1.0)
        max_val = kwargs.get('max_val', 1.0)
        
        for idx in indices:
            if isinstance(tensor, torch.Tensor):
                tensor[idx] = torch.tensor(random.uniform(min_val, max_val), 
                                          dtype=tensor.dtype, 
                                          device=tensor.device)
            else:  # numpy array
                tensor[idx] = random.uniform(min_val, max_val)
            
        return tensor
    
    def _inject_gaussian_noise(self, tensor: Union[torch.Tensor, np.ndarray], 
                              indices: List, 
                              **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入高斯噪声"""
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 0.1)
        
        for idx in indices:
            if isinstance(tensor, torch.Tensor):
                noise = torch.normal(mean=torch.tensor(mean), 
                                    std=torch.tensor(std)).to(tensor.device)
                tensor[idx] += noise
            else:  # numpy array
                noise = np.random.normal(loc=mean, scale=std)
                tensor[idx] += noise
            
        return tensor
    
    def _inject_zero_value(self, tensor: Union[torch.Tensor, np.ndarray], 
                          indices: List) -> Union[torch.Tensor, np.ndarray]:
        """注入零值错误"""
        for idx in indices:
            tensor[idx] = 0
            
        return tensor
    
    def _inject_constant(self, tensor: Union[torch.Tensor, np.ndarray], 
                        indices: List, 
                        **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入常量错误"""
        value = kwargs.get('value', 1.0)
        
        for idx in indices:
            tensor[idx] = value
            
        return tensor
    
    def _inject_scaling(self, tensor: Union[torch.Tensor, np.ndarray], 
                       indices: List, 
                       **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入缩放错误"""
        scale_factor = kwargs.get('scale_factor', 2.0)
        
        for idx in indices:
            tensor[idx] *= scale_factor
            
        return tensor
    
    def _inject_permutation(self, tensor: Union[torch.Tensor, np.ndarray], 
                           indices: List, 
                           **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """注入置换错误"""
        if len(indices) <= 1:
            return tensor
            
        # 提取所有索引位置的值
        values = [tensor[idx].item() if isinstance(tensor[idx], (torch.Tensor, np.ndarray)) else tensor[idx] 
                 for idx in indices]
        
        # 随机打乱
        random.shuffle(values)
        
        # 将打乱后的值放回
        for i, idx in enumerate(indices):
            tensor[idx] = values[i]
            
        return tensor
    
    def clear_history(self):
        """清除注入历史"""
        self.injection_history = []
    
    def get_injection_stats(self) -> Dict:
        """获取注入统计信息"""
        if not self.injection_history:
            return {"total_injections": 0}
            
        stats = {
            "total_injections": len(self.injection_history),
            "fault_types": {},
            "fault_patterns": {}
        }
        
        for record in self.injection_history:
            ft = record['fault_type'].value
            fp = record['fault_pattern'].value
            
            if ft in stats["fault_types"]:
                stats["fault_types"][ft] += 1
            else:
                stats["fault_types"][ft] = 1
                
            if fp in stats["fault_patterns"]:
                stats["fault_patterns"][fp] += 1
            else:
                stats["fault_patterns"][fp] = 1
        
        return stats


class BitFlipInjector(FaultInjector):
    """位翻转错误注入器"""
    
    def __init__(self, bit_position: Optional[int] = None, num_bits: int = 1, **kwargs):
        """
        初始化位翻转错误注入器
        
        Args:
            bit_position: 要翻转的位位置，如果为None则随机选择
            num_bits: 要翻转的位数量
            **kwargs: 其他参数
        """
        super().__init__(fault_type=FaultType.BIT_FLIP, **kwargs)
        self.bit_position = bit_position
        self.num_bits = num_bits
    
    def inject(self, tensor, indices=None, **kwargs):
        """注入位翻转错误"""
        kwargs['bit_position'] = kwargs.get('bit_position', self.bit_position)
        kwargs['num_bits'] = kwargs.get('num_bits', self.num_bits)
        return super().inject(tensor, indices, **kwargs)


class RandomValueInjector(FaultInjector):
    """随机值错误注入器"""
    
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, **kwargs):
        """
        初始化随机值错误注入器
        
        Args:
            min_val: 随机值下限
            max_val: 随机值上限
            **kwargs: 其他参数
        """
        super().__init__(fault_type=FaultType.RANDOM_VALUE, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
    
    def inject(self, tensor, indices=None, **kwargs):
        """注入随机值错误"""
        kwargs['min_val'] = kwargs.get('min_val', self.min_val)
        kwargs['max_val'] = kwargs.get('max_val', self.max_val)
        return super().inject(tensor, indices, **kwargs)


class GaussianNoiseInjector(FaultInjector):
    """高斯噪声错误注入器"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1, **kwargs):
        """
        初始化高斯噪声错误注入器
        
        Args:
            mean: 噪声均值
            std: 噪声标准差
            **kwargs: 其他参数
        """
        super().__init__(fault_type=FaultType.GAUSSIAN_NOISE, **kwargs)
        self.mean = mean
        self.std = std
    
    def inject(self, tensor, indices=None, **kwargs):
        """注入高斯噪声错误"""
        kwargs['mean'] = kwargs.get('mean', self.mean)
        kwargs['std'] = kwargs.get('std', self.std)
        return super().inject(tensor, indices, **kwargs)


class ZeroValueInjector(FaultInjector):
    """零值错误注入器"""
    
    def __init__(self, **kwargs):
        """初始化零值错误注入器"""
        super().__init__(fault_type=FaultType.ZERO_VALUE, **kwargs)


class ConstantInjector(FaultInjector):
    """常量错误注入器"""
    
    def __init__(self, value: float = 1.0, **kwargs):
        """
        初始化常量错误注入器
        
        Args:
            value: 要注入的常量值
            **kwargs: 其他参数
        """
        super().__init__(fault_type=FaultType.CONSTANT, **kwargs)
        self.value = value
    
    def inject(self, tensor, indices=None, **kwargs):
        """注入常量错误"""
        kwargs['value'] = kwargs.get('value', self.value)
        return super().inject(tensor, indices, **kwargs)


class ScalingInjector(FaultInjector):
    """缩放错误注入器"""
    
    def __init__(self, scale_factor: float = 2.0, **kwargs):
        """
        初始化缩放错误注入器
        
        Args:
            scale_factor: 缩放因子
            **kwargs: 其他参数
        """
        super().__init__(fault_type=FaultType.SCALING, **kwargs)
        self.scale_factor = scale_factor
    
    def inject(self, tensor, indices=None, **kwargs):
        """注入缩放错误"""
        kwargs['scale_factor'] = kwargs.get('scale_factor', self.scale_factor)
        return super().inject(tensor, indices, **kwargs)


class PermutationInjector(FaultInjector):
    """置换错误注入器"""
    
    def __init__(self, **kwargs):
        """初始化置换错误注入器"""
        super().__init__(fault_type=FaultType.PERMUTATION, **kwargs)
        # 置换需要多个点，设置为MULTIPLE模式
        self.fault_pattern = kwargs.get('fault_pattern', FaultPattern.MULTIPLE)


# 辅助函数

def float_to_binary(num):
    """将浮点数转换为二进制字符串"""
    # 使用IEEE 754格式
    if isinstance(num, float):
        # 针对Python float (C double, 通常是64位)
        import struct
        # 将float打包为8字节，然后转为二进制字符串
        return ''.join(f'{byte:08b}' for byte in struct.pack('>d', num))
    else:
        # 针对整数
        return bin(num)[2:]  # 去掉'0b'前缀

def binary_to_float(binary):
    """将二进制字符串转换为浮点数"""
    import struct
    # 将二进制字符串转为字节
    byte_length = len(binary) // 8
    bytes_data = int(binary, 2).to_bytes(byte_length, byteorder='big')
    
    if byte_length == 8:  # 64位双精度
        return struct.unpack('>d', bytes_data)[0]
    elif byte_length == 4:  # 32位单精度
        return struct.unpack('>f', bytes_data)[0]
    else:
        raise ValueError(f"Unsupported binary length: {len(binary)}")

def get_random_indices(tensor_shape, num_indices=1):
    """获取随机索引"""
    indices = []
    for _ in range(num_indices):
        idx = []
        for dim_size in tensor_shape:
            idx.append(random.randint(0, dim_size - 1))
        indices.append(tuple(idx))
    return indices

def create_pattern_indices(tensor_shape, pattern_type='checkerboard'):
    """创建特定模式的索引"""
    indices = []
    
    if pattern_type == 'checkerboard':
        # 棋盘格模式
        if len(tensor_shape) < 2:
            return [(i,) for i in range(0, tensor_shape[0], 2)]
        
        for i in range(tensor_shape[0]):
            for j in range((i % 2), tensor_shape[1], 2):
                indices.append((i, j))
                
    elif pattern_type == 'border':
        # 边界模式
        if len(tensor_shape) < 2:
            return [(0,), (tensor_shape[0]-1,)]
        
        # 上边界
        for j in range(tensor_shape[1]):
            indices.append((0, j))
        # 下边界
        for j in range(tensor_shape[1]):
            indices.append((tensor_shape[0]-1, j))
        # 左边界
        for i in range(1, tensor_shape[0]-1):
            indices.append((i, 0))
        # 右边界
        for i in range(1, tensor_shape[0]-1):
            indices.append((i, tensor_shape[1]-1))
            
    elif pattern_type == 'diagonal':
        # 对角线模式
        if len(tensor_shape) < 2:
            return [(i,) for i in range(tensor_shape[0])]
        
        min_dim = min(tensor_shape[0], tensor_shape[1])
        for i in range(min_dim):
            indices.append((i, i))
            
    return indices 