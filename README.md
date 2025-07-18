# SDC检测系统

基于算法基础容错(ABFT)的静默数据损坏(SDC)检测系统。该项目实现了简单的注意力机制(Attention)和多层感知机(MLP)模型，并使用ABFT方法检测计算过程中的数据损坏。

## 项目结构

```
sdc_check/
├── models/             # 模型实现
│   ├── __init__.py
│   ├── attention.py    # 注意力机制实现
│   └── mlp.py          # 多层感知机实现
├── abft/               # ABFT检测实现
│   ├── __init__.py
│   ├── detector.py     # SDC检测器
│   ├── utils.py        # 辅助功能
│   └── fault_injection.py # 错误注入模块
├── tests/              # 测试代码
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_mlp.py
│   └── test_fault_injection.py # 错误注入测试
├── main.py             # 主程序
└── README.md           # 项目说明
```

## 功能特性

1. **模型实现**
   - 自注意力机制(Self-Attention)
   - 多层感知机(MLP)

2. **ABFT检测功能**
   - 校验和计算与验证
   - 错误注入与检测
   - 损坏位置定位

3. **SDC检测器**
   - 通用张量校验和计算
   - 支持PyTorch和NumPy数据类型
   - 故障注入测试

4. **错误注入模块**
   - 多种错误类型：位翻转、随机值、高斯噪声、零值、常量、缩放、置换等
   - 灵活的注入位置策略：随机、指定位置、模式选择、块区域、结构化等
   - 多种错误模式：单点、多点、整行、整列、对角线、十字形、随机块等
   - 可扩展的错误注入器框架

## 依赖项

- Python 3.6+
- PyTorch 1.7+
- NumPy

## 安装

1. 克隆仓库
```bash
git clone git@github.com:cainiaogoroad/sdc_check.git
cd sdc_check
```

2. 安装依赖
```bash
pip install torch numpy
```

## 使用方法

### 运行演示程序

```bash
# 运行所有演示
python main.py

# 仅运行注意力机制演示
python main.py --model attention

# 仅运行MLP演示
python main.py --model mlp

# 仅运行SDC检测器演示
python main.py --model detector

# 仅运行错误注入器演示
python main.py --model injectors

# 使用自定义维度
python main.py --dim 64

# 不注入故障
python main.py --no-fault
```

### 运行测试

```bash
# 测试注意力机制
python tests/test_attention.py

# 测试MLP
python tests/test_mlp.py

# 测试错误注入模块
python tests/test_fault_injection.py
```

## ABFT方法

本项目使用算法基础容错(ABFT)方法检测静默数据损坏(SDC)：

1. **注意力机制ABFT**
   - 记录注意力权重矩阵
   - 计算行和与列和作为校验和
   - 验证阶段重新计算校验和并比较

2. **MLP的ABFT**
   - 记录线性层输出和激活函数后的张量
   - 计算多维度校验和
   - 基于校验和验证数据完整性

3. **通用SDC检测器**
   - 支持任意张量的校验和计算
   - 提供多种校验指标（总和、均值、标准差等）
   - 灵活的验证机制和容错阈值

## 错误注入框架

本项目实现了一个灵活的错误注入框架：

1. **错误类型(FaultType)**
   - `BIT_FLIP`: 位翻转错误
   - `RANDOM_VALUE`: 随机值错误
   - `GAUSSIAN_NOISE`: 高斯噪声
   - `ZERO_VALUE`: 值置零
   - `CONSTANT`: 常量替换
   - `SCALING`: 缩放错误
   - `PERMUTATION`: 元素置换

2. **注入位置策略(FaultLocation)**
   - `RANDOM`: 随机位置
   - `SPECIFIED`: 指定位置
   - `PATTERN`: 按模式选择
   - `BLOCK`: 块区域
   - `STRUCTURED`: 结构化选择

3. **错误模式(FaultPattern)**
   - `SINGLE`: 单点错误
   - `MULTIPLE`: 多点错误
   - `ROW`: 整行错误
   - `COLUMN`: 整列错误
   - `DIAGONAL`: 对角线错误
   - `CROSS`: 十字形错误
   - `RANDOM_BLOCK`: 随机块错误

4. **特定错误注入器**
   - `BitFlipInjector`: 位翻转错误注入器
   - `RandomValueInjector`: 随机值错误注入器
   - `GaussianNoiseInjector`: 高斯噪声错误注入器
   - `ZeroValueInjector`: 零值错误注入器
   - `ConstantInjector`: 常量错误注入器
   - `ScalingInjector`: 缩放错误注入器
   - `PermutationInjector`: 置换错误注入器

### 使用示例

```python
from abft.fault_injection import BitFlipInjector, FaultPattern, FaultLocation

# 创建位翻转错误注入器
injector = BitFlipInjector(
    num_bits=2,  # 翻转2个位
    fault_pattern=FaultPattern.SINGLE,  # 单点错误
    fault_location=FaultLocation.RANDOM  # 随机位置
)

# 在张量中注入错误
corrupted_tensor = injector.inject(original_tensor)
```

## 许可证

MIT

## 贡献

欢迎提交问题和贡献代码！ 