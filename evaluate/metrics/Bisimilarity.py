from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ot
from aalpy.automata import Mdp, MdpState

from .Metric import Metric
from evaluate.utils import print_table
from evaluate.utils.log import logger
from HierarchicMDP import Input


class Bisimilarity(Metric, ABC):
    ERROR_THRESHOLD_DEFAULT = 0.001

    def calculate(self, discounting_factor, error_threshold=ERROR_THRESHOLD_DEFAULT):
        original_mdp: Mdp = self.original_mdp
        if original_mdp is not None:
            logger.evaluating(f"Calculating {type(self).__name__} ...")
            orig_vs_orig = self.bisimilarity(original_mdp, original_mdp, discounting_factor, error_threshold)
            orig_vs_hal = self.bisimilarity(original_mdp, self.learned_model_hal, discounting_factor, error_threshold)
            orig_vs_alg = self.bisimilarity(original_mdp, self.learned_model_alg, discounting_factor, error_threshold)

            self.print(orig_vs_orig, orig_vs_hal, orig_vs_alg)

            return {"HAL": orig_vs_hal,
                    self.algorithm_name: orig_vs_alg}

    @classmethod
    def bisimilarity(cls, mdp1: Mdp, mdp2: Mdp, discounting_factor, error_threshold):
        distance_matrix = cls.calculate_distance_matrix(mdp1, mdp2, discounting_factor, error_threshold)
        return 1. - distance_matrix[mdp1.states.index(mdp1.initial_state)][mdp2.states.index(mdp2.initial_state)]

    @classmethod
    def calculate_distance_matrix(cls, mdp1: Mdp, mdp2: Mdp, discounting_factor, error_threshold):
        delta = 2 * error_threshold
        d_current = np.array([[(0. if state1.output == state2.output else 1.) for state2 in mdp2.states] for state1 in mdp1.states])  # access as d[mdp1.state][mdp2.state]
        d_old = d_current.copy()

        while delta > error_threshold:
            for i, state1 in enumerate(mdp1.states):
                for j, state2 in enumerate(mdp2.states):
                    if state1.output != state2.output:
                        continue

                    distance_per_input = {}
                    for a in mdp1.get_input_alphabet():  #assume same input alphabets
                        p_s1_a = Bisimilarity.get_transition_distribution(mdp1, state1, a)
                        p_s2_a = Bisimilarity.get_transition_distribution(mdp2, state2, a)
                        distance_per_input[a] = ot.emd2(p_s1_a, p_s2_a, M=d_old)

                    distance = discounting_factor * cls.distance_function(list(distance_per_input.values()))
                    d_current[i][j] = distance

            delta = np.average(np.abs(d_current - d_old))
            d_old = d_current.copy()

        return d_current

    @staticmethod
    def get_transition_distribution(mdp: Mdp, state: MdpState, input: Input):
        state_transitions = defaultdict(lambda: 0., {s: p for s, p in state.transitions[input]})
        return [state_transitions[s] for s in mdp.states]

    @staticmethod
    @abstractmethod
    def distance_function(distances_list: list[float]):
        pass

    def print(self, orig_vs_orig, orig_vs_hal, orig_vs_alg):
        logger.info(f"-------------------- {self.name} --------------------")
        logger.info(f"Bisimilarity of <x> with the original MDP")
        print_table([
            ["Original", "Hierarchical MDP", self.algorithm_name],
            [orig_vs_orig, orig_vs_hal, orig_vs_alg]
        ])


class BisimilarityMean(Bisimilarity):
    def calculate(self, discounting_factor_mean):
        return super().calculate(discounting_factor_mean)

    @staticmethod
    def distance_function(distances_list: list[float]):
        return np.mean(distances_list)


class BisimilarityMax(Bisimilarity):
    def calculate(self, discounting_factor_max):
        return super().calculate(discounting_factor_max)

    @staticmethod
    def distance_function(distances_list: list[float]):
        return np.max(distances_list)