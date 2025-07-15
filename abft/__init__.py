"""
算法基础容错(ABFT)检测静默数据损坏(SDC)的实现
"""

from .detector import SDCDetector
from .utils import checksum, verify_checksum

__all__ = ["SDCDetector", "checksum", "verify_checksum"] 