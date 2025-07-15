"""
测试MLP及其ABFT检测功能
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from abft.detector import SDCDetector

def test_mlp_forward():
    """测试MLP的前向传播"""
    # 创建一个小的测试输入
    batch_size, dim = 2, 32
    x = torch.randn(batch_size, dim)
    
    # 创建MLP模型
    mlp = MLP(in_features=dim, hidden_features=64, out_features=dim)
    
    # 前向传播
    output = mlp(x)
    
    # 检查输出形状
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"
    
    print("MLP前向传播测试通过")
    return True

def test_mlp_abft():
    """测试MLP的ABFT检测功能"""
    # 创建一个小的测试输入
    batch_size, dim = 2, 32
    x = torch.randn(batch_size, dim)
    
    # 创建MLP模型
    mlp = MLP(in_features=dim, hidden_features=64, out_features=dim)
    
    # 前向传播（启用ABFT）
    output = mlp(x, enable_abft=True)
    
    # 保存第一层输出的副本
    original_fc1_output = mlp.last_fc1_output.clone()
    
    # 验证完整性（此时应该没有错误）
    result = mlp.verify_integrity(layer='fc1')
    assert not result['is_corrupted'], "检测到错误的SDC"
    
    # 人为注入错误
    corrupted_output = original_fc1_output.clone()
    # 修改第一个批次的第一个特征
    corrupted_output[0, 0] += 0.1
    
    # 重新验证完整性（此时应该检测到错误）
    result = mlp.verify_integrity(layer='fc1', current_tensor=corrupted_output)
    assert result['is_corrupted'], "未能检测到人为注入的错误"
    
    print("MLP ABFT检测测试通过")
    return True

def test_mlp_sdc_detector():
    """测试使用SDCDetector检测MLP中的数据损坏"""
    # 创建一个小的测试输入
    batch_size, dim = 2, 32
    x = torch.randn(batch_size, dim)
    
    # 创建MLP模型和检测器
    mlp = MLP(in_features=dim, hidden_features=64, out_features=dim)
    detector = SDCDetector()
    
    # 前向传播
    output = mlp(x)
    
    # 注册MLP输出到检测器
    detector.register_tensor('mlp_output', output)
    
    # 验证（此时应该没有错误）
    result = detector.verify_tensor('mlp_output')
    assert not result['is_corrupted'], "检测到错误的SDC"
    
    # 注入错误
    corrupted_output = detector.inject_fault('mlp_output', magnitude=0.1)
    
    # 重新验证（此时应该检测到错误）
    result = detector.verify_tensor('mlp_output', corrupted_output)
    assert result['is_corrupted'], "未能检测到人为注入的错误"
    
    print("SDCDetector检测MLP数据损坏测试通过")
    return True

if __name__ == "__main__":
    # 运行所有测试
    test_mlp_forward()
    test_mlp_abft()
    test_mlp_sdc_detector()
    
    print("所有测试通过!") 