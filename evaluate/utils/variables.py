import argparse
from collections.abc import Callable
from enum import Enum

from evaluate.metrics import *
from evaluate.utils.gsm import *


class TargetEnum(Enum):
    pass


class GeneralTarget(TargetEnum):
    Learning = "Learning"
    Evaluation = "Evaluation"

    @classmethod
    def _target_order(cls):
        return 0

    @classmethod
    def _description(cls):
        return "General"


class LearningAlgorithms(TargetEnum):
    IOAlergia = IOAlergia
    LikelihoodRatio = LikelihoodRatio
    AkaikeInformationCriterion = AkaikeInformationCriterion

    @classmethod
    def _target_order(cls):
        return 1

    @classmethod
    def _description(cls):
        return "Learning Algorithm"


class Metrics(TargetEnum):
    BisimilarityMax = BisimilarityMax
    BisimilarityMean = BisimilarityMean
    ComplianceStepNormalized = ComplianceStepNormalized
    ComplianceNotStepNormalized = ComplianceNotStepNormalized

    @classmethod
    def _target_order(cls):
        return 2

    @classmethod
    def _description(cls):
        return "Metrics"


class MDPTarget(TargetEnum):
    Individual = "Individual MDPs"
    Combined = "Combined MDP"

    @classmethod
    def _target_order(cls):
        return 3

    @classmethod
    def _description(cls):
        return "MDP creation"


class TracesTarget(TargetEnum):
    Generation = "Trace Generation"

    @classmethod
    def _target_order(cls):
        return 4

    @classmethod
    def _description(cls):
        return "Trace generation"


class VariableTarget:
    MDP = MDPTarget
    METRIC = Metrics
    ALGORITHM = LearningAlgorithms
    TRACES = TracesTarget
    GENERAL = GeneralTarget


@dataclass
class Variable:
    abbreviation: str
    type: type
    default: Any
    description: str
    target: TargetEnum
    choices: tuple = None
    name: str = None  # added later
    exact_description: str = None
    only_relevant_if: Callable[[argparse.Namespace], bool] = None


@dataclass
class PlottableVariable(Variable):
    pass


@dataclass
class StaticVariable(Variable):
    num_choices: str = "single"


def arg_true(arg_name: str, args: argparse.Namespace):
    return bool(getattr(args, arg_name))

def arg_false(arg_name: str, args: argparse.Namespace, ):
    return not arg_true(arg_name, args)