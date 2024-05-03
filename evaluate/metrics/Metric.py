from abc import ABC, abstractmethod

from aalpy.automata import Mdp


class Metric(ABC):
    """ Abstract base class for metrics."""
    def __init__(self, original_mdp: Mdp, learned_model_hal: Mdp, learned_model_alg: Mdp, algorithm_name: str):
        """
        :param original_mdp             the original MDP to be learned
        :param learned_model_hal        the model learned with hierarchical-automata-learning
        :param learned_model_alg        the model learned with another algorithm
        :param algorithm_name           the name of the other algorithm, for displaying
        """
        self.original_mdp: Mdp = original_mdp
        self.learned_model_hal = learned_model_hal
        self.learned_model_alg = learned_model_alg
        self.algorithm_name = algorithm_name

    @abstractmethod
    def calculate(self) -> dict[str, float]:
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def print_description(cls):
        # print(f"{cls.__name__}:")
        print(cls.__doc__)
