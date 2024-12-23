# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass


@dataclass
class NoisyOrConfig:
    p_low: float = 0.5
    p_high: float = 1.0
    q_low: float = 0.5


@dataclass
class DiscreteNNConfig:
    input_dim: int
    num_hidden_layers: int = 3
    nnodes_per_layer: int = 1000
    nonlinearity: str = "relu"
    min_strength: float = 0.5
    max_strength: float = 1
    density: float = 0.99
    input_alphabets: list = None
    output_alphabet: list = None


@dataclass
class SyntheticConfig:
    num_layers: int = 2
    nodes_per_layer: int = 4
    density: float = 1.0
    
    default_alphabet_size: int = 10
    output_parents_alphabet_size: int = 10
    output_alphabet_size: int = 10
    
    noisy_or_config: NoisyOrConfig = NoisyOrConfig()
    discrete_nn_config: DiscreteNNConfig = DiscreteNNConfig(1)
    
    @classmethod
    def from_dict(cls, config_dict):
        config_dict["noisy_or_config"] = NoisyOrConfig(**config_dict["noisy_or_config"])
        config_dict["discrete_nn_config"] = DiscreteNNConfig(**config_dict["discrete_nn_config"])
        return SyntheticConfig(**config_dict)