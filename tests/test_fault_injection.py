"""
测试错误注入模块的各种功能
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abft.fault_injection import (
    FaultType, FaultLocation, FaultPattern, FaultTiming,
    FaultInjector, BitFlipInjector, RandomValueInjector, 
    GaussianNoiseInjector, ZeroValueInjector, ConstantInjector,
    ScalingInjector, PermutationInjector,
    get_random_indices, create_pattern_indices
)

def test_bit_flip_injector():
    """测试位翻转错误注入器"""
    print("\n测试位翻转错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建位翻转注入器
    injector = BitFlipInjector(
        fault_location=FaultLocation.SPECIFIED,
        fault_pattern=FaultPattern.SINGLE
    )
    
    # 在指定位置注入错误
    indices = (1, 1)  # 中心位置
    corrupted = injector.inject(tensor, indices=indices)
    
    print(f"注入位翻转错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_random_value_injector():
    """测试随机值错误注入器"""
    print("\n测试随机值错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建随机值注入器
    injector = RandomValueInjector(
        min_val=-5.0,
        max_val=5.0,
        fault_pattern=FaultPattern.MULTIPLE,
        fault_location=FaultLocation.RANDOM
    )
    
    # 注入多点随机错误
    corrupted = injector.inject(tensor, num_faults=3)
    
    print(f"注入随机值错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_gaussian_noise_injector():
    """测试高斯噪声错误注入器"""
    print("\n测试高斯噪声错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建高斯噪声注入器
    injector = GaussianNoiseInjector(
        mean=0.0,
        std=0.5,
        fault_pattern=FaultPattern.ROW
    )
    
    # 注入整行高斯噪声
    corrupted = injector.inject(tensor)
    
    print(f"注入高斯噪声后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_zero_value_injector():
    """测试零值错误注入器"""
    print("\n测试零值错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建零值注入器
    injector = ZeroValueInjector(
        fault_pattern=FaultPattern.CROSS
    )
    
    # 注入十字形零值错误
    corrupted = injector.inject(tensor)
    
    print(f"注入零值错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_constant_injector():
    """测试常量错误注入器"""
    print("\n测试常量错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建常量注入器
    injector = ConstantInjector(
        value=-999.0,
        fault_pattern=FaultPattern.DIAGONAL
    )
    
    # 注入对角线常量错误
    corrupted = injector.inject(tensor)
    
    print(f"注入常量错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_scaling_injector():
    """测试缩放错误注入器"""
    print("\n测试缩放错误注入器...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    print(f"原始张量:\n{tensor}")
    
    # 创建缩放注入器
    injector = ScalingInjector(
        scale_factor=10.0,
        fault_pattern=FaultPattern.RANDOM_BLOCK
    )
    
    # 注入随机块缩放错误
    corrupted = injector.inject(tensor, block_size=(2, 2))
    
    print(f"注入缩放错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_permutation_injector():
    """测试置换错误注入器"""
    print("\n测试置换错误注入器...")
    
    # 创建一个测试张量，每个元素值不同
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    print(f"原始张量:\n{tensor}")
    
    # 创建置换注入器
    injector = PermutationInjector(
        fault_location=FaultLocation.PATTERN
    )
    
    # 创建一个模式索引
    indices = create_pattern_indices(tensor.shape, pattern_type='diagonal')
    
    # 注入置换错误
    corrupted = injector.inject(tensor, indices=indices)
    
    print(f"注入置换错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_custom_pattern():
    """测试自定义错误模式"""
    print("\n测试自定义错误模式...")
    
    # 创建一个测试张量
    tensor = torch.ones(5, 5)
    print(f"原始张量形状: {tensor.shape}")
    
    # 创建自定义模式
    pattern_indices = create_pattern_indices(tensor.shape, pattern_type='border')
    print(f"边界模式索引数量: {len(pattern_indices)}")
    
    # 创建随机值注入器
    injector = RandomValueInjector(
        min_val=-1.0,
        max_val=1.0,
        fault_location=FaultLocation.SPECIFIED
    )
    
    # 注入边界模式错误
    corrupted = injector.inject(tensor, indices=pattern_indices)
    
    print(f"注入边界模式错误后的张量:\n{corrupted}")
    print(f"是否发生变化: {torch.any(tensor != corrupted).item()}")
    
    return True

def test_probability():
    """测试错误注入概率"""
    print("\n测试错误注入概率...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    
    # 创建低概率注入器
    injector = RandomValueInjector(
        probability=0.1,  # 10%的概率注入错误
        fault_pattern=FaultPattern.SINGLE
    )
    
    # 多次尝试注入
    count_changed = 0
    num_trials = 100
    
    for i in range(num_trials):
        corrupted = injector.inject(tensor.clone())
        if torch.any(corrupted != tensor):
            count_changed += 1
    
    print(f"设置注入概率为10%，在{num_trials}次尝试中发生注入的次数: {count_changed}")
    print(f"实际注入概率: {count_changed / num_trials:.2f}")
    
    return True

def test_injection_history():
    """测试注入历史记录"""
    print("\n测试注入历史记录...")
    
    # 创建一个测试张量
    tensor = torch.ones(3, 3)
    
    # 创建注入器
    injector = FaultInjector()
    
    # 多次注入不同类型的错误
    injector.fault_type = FaultType.RANDOM_VALUE
    injector.inject(tensor.clone())
    
    injector.fault_type = FaultType.ZERO_VALUE
    injector.inject(tensor.clone())
    
    injector.fault_type = FaultType.SCALING
    injector.inject(tensor.clone())
    
    # 获取注入统计信息
    stats = injector.get_injection_stats()
    print(f"注入统计信息:\n{stats}")
    
    # 清除历史
    injector.clear_history()
    stats_after_clear = injector.get_injection_stats()
    print(f"清除历史后的统计信息:\n{stats_after_clear}")
    
    return True

def test_numpy_support():
    """测试NumPy数组支持"""
    print("\n测试NumPy数组支持...")
    
    # 创建一个NumPy测试数组
    array = np.ones((3, 3))
    print(f"原始NumPy数组:\n{array}")
    
    # 创建注入器
    injector = RandomValueInjector(
        min_val=-5.0,
        max_val=5.0,
        fault_pattern=FaultPattern.MULTIPLE
    )
    
    # 注入错误
    corrupted = injector.inject(array, num_faults=3)
    
    print(f"注入错误后的NumPy数组:\n{corrupted}")
    print(f"是否发生变化: {np.any(array != corrupted)}")
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("错误注入模块测试")
    print("="*50)
    
    # 运行所有测试
    test_bit_flip_injector()
    test_random_value_injector()
    test_gaussian_noise_injector()
    test_zero_value_injector()
    test_constant_injector()
    test_scaling_injector()
    test_permutation_injector()
    test_custom_pattern()
    test_probability()
    test_injection_history()
    test_numpy_support()
    
    print("\n所有测试完成!") 