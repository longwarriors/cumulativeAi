"""
数据预处理模块

负责对原始数据进行清洗、转换和预处理，包括：
- 缺失值处理（数值型和分类型）
- 分类变量编码（LabelEncoder）
- 异常值和无穷大值处理
- 数据类型转换优化

支持训练集和测试集的一致性处理。
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')  # numerical imputation strategy
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')  # categorical imputation strategy
        self.standard_scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df: pd.DataFrame, is_train=True) -> pd.DataFrame:
        """
        对数据进行预处理，包括缺失值处理、编码和标准化。

        :param df: 输入待处理的DataFrame
        :param is_train: 是否为训练集
        :return: 预处理后的DataFrame
        """
        logger.info(f"开始预处理{'训练集' if is_train else '测试集'}数据，形状: {df.shape}")
        df_copy = df.copy()
        num_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df_copy.select_dtypes(include=['object']).columns

        # 排除目标列和ID列
        exclude_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']
        num_cols = [col for col in num_cols if col not in exclude_cols]
        cat_cols = [col for col in cat_cols if col not in exclude_cols]

        logger.info("处理无穷大值")
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

        logger.info(f"处理数值型特征的缺失: {len(num_cols)}列")
        if is_train:
            df_copy[num_cols] = self.num_imputer.fit_transform(df_copy[num_cols])
        else:
            df_copy[num_cols] = self.num_imputer.transform(df_copy[num_cols])

        logger.info(f"处理分类特征的缺失: {len(cat_cols)}列")
        if is_train:
            df_copy[cat_cols] = self.cat_imputer.fit_transform(df_copy[cat_cols])
        else:
            df_copy[cat_cols] = self.cat_imputer.transform(df_copy[cat_cols])

        logger.info(f"编码类别变量：")
        for col in cat_cols:
            if df_copy[col].nunique() > 2:  # 多于两个类别的变量使用独热编码
                df_copy = pd.get_dummies(df_copy, columns=[col], prefix=col, dummy_na=False)
            else:  # 二分类变量使用标签编码
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col])
                else:  # 处理测试集中可能出现的新类别
                    if col in self.label_encoders:
                        df_copy[col] = df_copy[col].map(
                            lambda x: x if x in self.label_encoders[col].classes_ else 'missing'
                        )
                        df_copy[col] = self.label_encoders[col].transform(df_copy[col])
                    else:  # 如果训练集中没有这个列，跳过
                        logger.warning(f"列 {col} 在训练集中不存在，跳过编码")

        logger.info("预处理完成")
        return df_copy


class AdvancedPreprocessor:
    """
    扩展的预处理器，利用 df.pipe() 提升代码可读性和流程清晰度。
    """

    def __init__(self,
                 numerical_imputation_strategy: str = 'median',
                 categorical_imputation_strategy: str = 'constant',
                 categorical_fill_value: str = 'missing',
                 exclude_cols: Optional[List[str]] = None,
                 target_col: str = 'TARGET'):
        """
        初始化高级预处理器。
        :param numerical_imputation_strategy:数值特征缺失值填充策略，如'median', 'mean'等。
        :param categorical_imputation_strategy:分类特征缺失值填充策略，如'constant', 'most_frequent'等。
        :param categorical_fill_value:分类特征常数填充值。
        :param exclude_cols:需要从预处理中排除的ID列名列表。
        :param target_col:目标列名，也需要从预处理中排除。
        """
        self.numerical_imputation_strategy = numerical_imputation_strategy
        self.categorical_imputation_strategy = categorical_imputation_strategy
        self.categorical_fill_value = categorical_fill_value
        self.exclude_cols = set(exclude_cols or ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'])
        self.target_col = target_col

        # 初始化组件
        self.num_imputer = SimpleImputer(strategy=self.numerical_imputation_strategy)
        self.cat_imputer = SimpleImputer(strategy=self.categorical_imputation_strategy,
                                         fill_value=self.categorical_fill_value)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()

        # 存储训练时的列信息
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.onehot_cols: List[str] = []
        self.label_encoder_cols: List[str] = []
        self.feature_names_: List[str] = []

        # 标记是否已经fit过
        self.is_fitted = False

    def _handle_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        替换无穷大和无穷小值为NaN。
        """
        logger.info("处理无穷大值...")
        return df.replace([np.inf, -np.inf], np.nan)

    def _impute_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数值特征进行缺失值填充。
        """
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [col for col in num_cols if col not in self.exclude_cols]
        if not num_cols:
            logger.warning("没有数值型特征需要填充。")
            return df
        logger.info(f"处理数值型特征的缺失: {len(num_cols)}列")
        if self.is_train:
            df[self.num_cols] = self.num_imputer.transform(df[self.num_cols])
        else:
            df[self.num_cols] = self.num_imputer.fit_transform(df[self.num_cols])
        return df

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        预处理全流程。
        :param df:要预处理的数据。
        :param is_train:是否为训练数据。
        :return:预处理后的数据。
        """
        logger.info(f"开始预处理{'训练' if is_train else '测试'}数据")
        # 将ID列和TARGET列暂时保存，并在所有转换完成后重新合并
        preserved_cols = df[list(self.exclude_cols.intersection(df.columns))].copy()
        df_working = df.drop(columns=list(self.exclude_cols.intersection(df.columns)), errors='ignore').copy()
        df_processed = df_working.pipe(self._handle_infinities).pipe(self._impute_numerical)


if __name__ == "__main__":
    data_train = {
        'SK_ID_CURR': [1, 2, 3, 4, 5, 6],
        'Feature_Num1': [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
        'Feature_Num2': [1.0, np.inf, 3.0, 4.0, -np.inf, 6.0],
        'Feature_Cat1': ['A', 'B', 'A', 'C', 'B', 'A'],  # High-cardinality categorical
        'Feature_Cat2': ['Yes', 'No', 'Yes', 'No', np.nan, 'Yes'],  # Binary categorical
        'Feature_Cat3': ['X', 'Y', 'X', 'Z', 'Y', 'X'],  # Another high-cardinality
        'Some_Other_Col': ['val1', 'val2', 'val3', 'val4', 'val5', 'val6'],  # Untouched feature type
        'TARGET': [0, 1, 0, 1, 0, 1]
    }
    df_train = pd.DataFrame(data_train)

    data_test = {
        'SK_ID_CURR': [7, 8, 9, 10],
        'Feature_Num1': [70.0, np.nan, 90.0, 100.0],
        'Feature_Num2': [7.0, 8.0, np.inf, 10.0],
        'Feature_Cat1': ['B', 'D', 'C', 'A'],  # 'D' is new category
        'Feature_Cat2': ['No', 'Yes', 'No', 'Yes'],
        'Feature_Cat3': ['Y', 'A', 'Z', 'Y'],  # 'A' is new category
        'Some_Other_Col': ['val7', 'val8', 'val9', 'val10'],
        'TARGET': [0, 0, 1, 1]
    }
    df_test = pd.DataFrame(data_test)

    preprocessor = Preprocessor()
    print("\n--- 训练数据预处理 ---")
    df_train_processed = preprocessor.preprocess(df_train, is_train=True)
    print(df_train_processed.head())
    print(f"训练集处理后形状: {df_train_processed.shape}")
    print(df_train_processed.info(verbose=True, show_counts=True))
