import yaml, argparse
from config import FullConfig

def parse_args():
    parser = argparse.ArgumentParser(description='训练量子-经典混合transformer模型')

    # 添加CLI参数
    parser.add_argument('--config_file', type=str, default=None, help='YAML配置文件的路径')
    parser.add_argument('--num_steps', type=int, default=1000, help='训练的总步数')
    parser.add_argument('--batch_size', type=int, default=8, help='每个批次的样本数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--q_device', type=str, default="lightning.gpu", help='Pennylane后端设备')
    parser.add_argument('--use_quantum_ffn', type=bool, default=True, help='是否使用量子前馈网络')



def load_config(args: argparse.ArgumentParser) -> FullConfig:
    config = FullConfig() # 默认参数配置
    with open(path, 'r') as f:
        raw_dict = yaml.load(f)
    config_instance = FullConfig(
        # 子层也要分别实例化
        # FullConfig.training.__class__ = TrainingConfig
        training=FullConfig.training.__class__(**raw_dict['training']),
        # FullConfig.model.__class__ = ModelConfig
        model=FullConfig.model.__class__(**raw_dict['model']),
        # FullConfig.quantum.__class__ = QuantumConfig
        quantum=FullConfig.quantum.__class__(**raw_dict['quantum']),
        tokenizer_name=raw_dict['tokenizer_name']
    )
    return config_instance
