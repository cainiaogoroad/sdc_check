"""
算法基础容错(ABFT)检测静默数据损坏(SDC)的实现
"""

from .detector import SDCDetector
from .utils import checksum, verify_checksum
from .fault_injection import (
    FaultType, FaultLocation, FaultPattern, FaultTiming,
    FaultInjector, BitFlipInjector, RandomValueInjector, 
    GaussianNoiseInjector, ZeroValueInjector, ConstantInjector,
    ScalingInjector, PermutationInjector
)

__all__ = [
    "SDCDetector", "checksum", "verify_checksum",
    "FaultType", "FaultLocation", "FaultPattern", "FaultTiming",
    "FaultInjector", "BitFlipInjector", "RandomValueInjector", 
    "GaussianNoiseInjector", "ZeroValueInjector", "ConstantInjector",
    "ScalingInjector", "PermutationInjector"
] 