import math
import torch
import numpy as np

from baseline.baseAgent import baseAgent
from baseline.utils import jsonParser
from collections import namedtuple, deque
from typing import List, Tuple


MAXIMUM_FLOAT_VALUE = float("inf")
KnwonBounds = namedtuple("KnownBounds", ["min", "max"])
ModelOutput = namedtuple(
    "ModelOutput", ["value", "reward", "policy_logits", "hidden_state"]
)


class MinMaxStats:

    def __init__(self, known_bounds: KnwonBounds):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
    
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.maximum, value)
    
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (valuee - self.minimum) / (self.maximum - self.minimum)
        return value


class MurozeroConfig:
    def __init__(self, path):
        parser = jsonParser(path)
        self.data = parser.loadParser()

        for key, value in self.data.items():
            setattr(self, key, value)


class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0

        # key: action, value: Node
        self.children = {}

        self.hidden_state = None
        self.reward = 0
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        
        return self.value_sum / self.visit_count


class Game:
    def __init__(
        self,
        config: MurozeroConfig,
        action_space_size: int,
        discount: float
    ):