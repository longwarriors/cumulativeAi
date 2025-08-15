from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    num_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-4
    seq_len: int = 128
    tokenizer_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_subset: str = "wikitext-103-v1"
    train_spilt: str = "train[:1%]"
    valid_spilt: str = "validation"


@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    num_qubits: int = 16
    use_quantum_ffn: bool = True


@dataclass
class QuantumConfig:
    q_device: Literal["lightning.qubit", "lightning.gpu"] = "lightning.gpu"


@dataclass
class FullConfig:
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    quantum: QuantumConfig = QuantumConfig()

