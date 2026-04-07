"""
PatchTST용 데이터셋 (LSTM dataset과 동일한 구조 재사용)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm.dataset import AngleSequenceDataset  # noqa: F401

__all__ = ['AngleSequenceDataset']
