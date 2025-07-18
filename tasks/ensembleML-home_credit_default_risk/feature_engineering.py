"""https://gemini.google.com/app/c56e035010b22958
实现完整的特征工程pipeline，包括：

1. 辅助信息表的聚合特征：
    - 信贷局数据（bureau）
    - 信贷局余额数据（bureau_balance）
    - 历史申请数据（previous_application）
    - POS和现金贷款余额（POS_CASH_balance）
    - 信用卡余额数据（credit_card_balance）
    - 分期付款行为数据（installments_payments）

2. 领域知识构造特征：
    - 贷款金额与收入比率
    - 信用卡使用率
    - 历史逾期次数
    - 信贷局评分与贷款金额的关系

3. 交互特征：
    - 贷款金额与信用卡余额的交互特征
    - 信用卡使用率与历史逾期次数的交互特征
    - EXT_SOURCE特征组合（均值、标准差、乘积）
    - 重要特征间的乘积、比值、差值交互

4. 特征选择和转换：
    - 数值型特征的缺失值填充
    - 类别型特征的缺失值填充
    - 类别型特征的标签编码
    - 数值型特征的标准化
"""
from data_loader import CustomDataLoader, full_data_dict
import pandas as pd
import numpy as np
import logging, yaml

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, full_data_dict: dict):
        """
        初始化特征工程类，接收完整数据字典。

        :param full_data_dict: 包含所有数据集的字典
        """
        self.full_data_dict = full_data_dict

    def aggregate_bureau(self, df_main: pd.DataFrame) -> pd.DataFrame:
        """
        聚合信贷局数据（bureau）特征。
        :param df_main: 主表数据
        :return: 添加了信贷局聚合特征的DataFrame
        """
        logger.info("开始聚合信贷局数据（bureau表）的特征")
        if 'bureau' not in self.full_data_dict:
            logger.warning("信贷局数据（bureau表）不存在，跳过聚合")
            return df_main
        df_bureau = self.full_data_dict['bureau']

        # 数值列的聚合
        num_agg = {
            "DAYS_CREDIT": ["min", "max", "mean", "var"], # int64
            "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"], # float64
            "DAYS_CREDIT_UPDATE": ["mean"], # int64
            "CREDIT_DAY_OVERDUE": ["max", "mean"], # int64
            "CNT_CREDIT_PROLONG": ["sum"], # int64
            "AMT_CREDIT_MAX_OVERDUE": ["mean"], # float64
            "AMT_CREDIT_SUM": ["max", "mean", "sum"], # float64
            "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"], # float64
            "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"], # float64
            "AMT_CREDIT_SUM_OVERDUE": ["mean"], # float64
            "AMT_ANNUITY": ["max", "mean"], # float64
        }

        # 类别列的聚合（只计算计数）
        cat_agg = {cat: ["count"] for cat in df_bureau.select_dtypes(include=['object']).columns if cat != "SK_ID_BUREAU"}
        bureau_agg = df_bureau.groupby("SK_ID_CURR").agg({**num_agg, **cat_agg})
        bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]

        # 合并聚合特征到主表
        df_main = df_main.merge(bureau_agg, on="SK_ID_CURR", how="left")
        logger.info(f"信贷局数据（bureau表）聚合特征添加完成，添加了{bureau_agg.shape[1]}个特征")
        return df_main

    def aggregate_bureau_balance(self, df_main: pd.DataFrame) -> pd.DataFrame:
        """
        聚合信贷局余额数据（bureau_balance）特征。
        :param df_main: 主表数据
        :return: 添加了信贷局余额聚合特征的DataFrame
        """
        logger.info("开始聚合信贷局余额数据（bureau_balance表）的特征")
        if 'bureau_balance' not in self.full_data_dict:
            logger.warning("信贷局余额数据（bureau_balance表）不存在，跳过聚合")
            return df_main
        df_bureau_balance = self.full_data_dict['bureau_balance']

        # 按每个客户（由SK_ID_CURR标识）聚合
        bb_agg = {"MONTHS_BALANCE": ["min", "max", "mean"]},  # int64
        bureau_balance_agg = df_bureau_balance.groupby("SK_ID_BUREAU").agg(bb_agg)

        # 计算每个SK_ID_BUREAU的状态分布
        status_agg = df_bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts().unstack(fill_value=0)
        status_agg.columns = [f"BB_STATUS_{col}" for col in status_agg.columns]

        # 合并到主表
        df_bureau_balance = df_bureau_balance[['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates()
        df_bureau_balance = df_bureau_balance.merge(status_agg, on='SK_ID_BUREAU', how='left')

        # ���合到主表
        bureau_balance_agg = df_bureau_balance.groupby('SK_ID_CURR').sum()

        # 合并聚合特征到主表
        df_main = df_main.merge(bureau_balance_agg, on="SK_ID_CURR", how="left")
        logger.info(f"信贷局余额数据（bureau_balance表）聚合特征添加完成，添加了{bureau_balance_agg.shape[1]}个特征")
        return df_main

if __name__ == '__main__':
    # 分析信贷局数据
    df_bureau = full_data_dict.get('bureau')
    df_bureau_balance = full_data_dict.get('bureau_balance')
    logger.info(f"bureau表每列的数据类型为：\n{df_bureau.dtypes}\n")
    logger.info(f"bureau_balance表每列的数据类型为：\n{df_bureau_balance.dtypes}\n")
    # logger.info(f"bureau表的列名为：\n{df_bureau.columns.tolist()}\n")
    # logger.info(f"bureau表的统计信息为：\n{df_bureau.describe()}\n")
    # logger.info(f"bureau表的概览信息为：\n{df_bureau.head()}\n")