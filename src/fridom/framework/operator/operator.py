from __future__ import annotations

from abc import ABC, abstractmethod

# use generic type for input and output
from typing import TypeVar, Generic

T = TypeVar('T')


class Operator(ABC):

    @abstractmethod
    def __call__(self, x: T) -> T: ...

    def __add__(self, other: Operator) -> Operator:
        return 

class _AddOperator(Operator):
    def __init__(self, op1: Operator, op2: Operator) -> None:
        self.op1 = op1
        self.op2 = op2

    def __call__(self, x: T) -> T:
        return self.op2(x) + self.op1(x)

class _SubOperator(Operator):
    def __init__(self, op1: Operator, op2: Operator) -> None:
        self.op1 = op1
        self.op2 = op2

    def __call__(self, x: T) -> T:
        return self.op2(x) - self.op1(x)

class _ScalarMulOperator(Operator):
    def __init__(self, op: Operator, scalar: float) -> None:
        self.op = op
        self.scalar = scalar

    def __call__(self, x: T) -> T:
        return self.scalar * self.op(x)

class _ComposeOperator(Operator):
    def __init__(self, op1: Operator, op2: Operator) -> None:
        self.op1 = op1
        self.op2 = op2

    def __call__(self, x: T) -> T:
        return self.op2(self.op1(x))