# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass


@dataclass
class Query:
    # P(Y = y | do(X = x))
    action: int  # X
    target: int  # Y
    action_value: int  # x
    target_value: int  # y