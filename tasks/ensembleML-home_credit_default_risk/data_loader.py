"""
数据加载模块，负责加载多源数据集，包括：
- application_train.csv: 主训练数据
- application_test.csv: 主测试数据
- bureau.csv: 信贷局数据
- bureau_balance.csv: 信贷局余额数据
- previous_application.csv: 历史申请数据
- POS_CASH_balance.csv: POS和现金贷款余额
- credit_card_balance.csv: 信用卡余额数据
- installments_payments.csv: 分期付款数据
"""
import pandas as pd
import numpy as np
import os, logging, yaml

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataLoader:
    """
    定制数据加载器类，支持自动数据验证和错误处理。
    """

    def __init__(self, configuration: dict):
        self.data_files = configuration.get('datafiles')
        self.dataframes = {}

    def load_data(self):
        """加载所有数据集"""
        for file_name, file_path in self.data_files.items():
            try:
                logger.info(f"加载数据集: {file_name}，路径: {file_path}")
                df = pd.read_csv(file_path)
                self.dataframes[file_name] = df
                logger.info(f"数据集 {file_name} 加载成功，形状: {df.shape}")
            except FileNotFoundError:
                logger.error(f"文件未找到: {file_path}")
            except pd.errors.EmptyDataError:
                logger.error(f"文件为空: {file_path}")
            except Exception as e:
                logger.error(f"加载 {file_name} 失败: {str(e)}")
        return self.dataframes

    def get_train_test_data(self) -> (pd.DataFrame, pd.DataFrame):
        """返回训练集和测试集"""
        if all(k in self.dataframes for k in ['application_train', 'application_test']):
            return self.dataframes["application_train"], self.dataframes["application_test"]
        else:
            logger.error("训练集或测试集数据未加载，请检查数据文件路径和名称")
            raise ValueError("请先调用load_data()方法加载数据")


# 加载所有数据集
with open('config.yaml', 'r', encoding='utf-8') as stream:
    config = yaml.safe_load(stream)
logger.info("加载全部数据集")
data_loader = CustomDataLoader(config)
full_data_dict = data_loader.load_data()
logger.info(f"多源数据集加载完成，数据集文件名包括: {list(full_data_dict.keys())}")

if __name__ == "__main__":
    train_df, test_df = data_loader.get_train_test_data()
    print(train_df.shape)
    print(test_df.shape)
