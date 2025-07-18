# https://claude.ai/chat/0be61e5a-bcba-444c-b778-79827cd4dbc7
from sklearn.impute import SimpleImputer
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Set


class ImputeStrategy(Enum):
    """填充策略枚举"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONST = "constant"


class EncodingStrategy(Enum):
    """编码策略枚举"""
    LABEL = "label"
    ONEHOT = "onehot


@dataclass
class PreprocessorConfig:
    """预处理配置数据类"""
    numerical_imputation_strategy: ImputeStrategy = ImputeStrategy.MEDIAN
    categorical_imputation_strategy: ImputeStrategy = ImputeStrategy.CONST
    categorical_fill_value: str = "missing"
    exclude_cols: Optional[Set[str]] = None
    target_col: str = "TARGET"
    encoding_strategy: EncodingStrategy = EncodingStrategy.LABEL

    def __post_init__(self):
        """exclude_cols是一个集合"""
        if self.exclude_cols is None:
            self.exclude_cols = {'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'}

@dataclass
class ColumnsInfo:
    """列信息数据类"""
    numerical_cols: List[str]
    categorical_cols: List[str]
    onehot_cols: List[str]
    label_encoder_cols: List[str]
    feature_names: List[str]

    def __post_init__(self):
        """初始化列信息"""
        self.numerical_cols = self.numerical_cols or []
        self.categorical_cols = self.categorical_cols or []
        self.onehot_cols = self.onehot_cols or []
        self.label_encoder_cols = self.label_encoder_cols or []
        self.feature_names = self.feature_names or []
