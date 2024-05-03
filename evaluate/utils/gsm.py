from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from aalpy.learning_algs.general_passive.GeneralizedStateMerging import GeneralizedStateMerging
import aalpy.learning_algs.general_passive.ScoreFunctionsGSM as ScoreFunctions


class ParameterizedLearningAlgorithm:
    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        return dict(
            output_behavior="moore",
            transition_behavior="stochastic",
            debug_lvl=2,
        )

    def get_GSM(self):
        def run_GSM(d):
            gsm = GeneralizedStateMerging(d, **self.params)
            gsm.run()
            return gsm.root

        return run_GSM

    def run_GSM(self, data):
        return self.get_GSM()(data)

    def __call__(self, data):
        params = self.params
        return self.run_GSM(data).to_automaton(params["output_behavior"], params["transition_behavior"])

    def __str__(self):
        # Could also use dataclass.__str__ but this avoids __qualname__ which can be annoying when using functions
        return f"{self.__class__.__name__}({', '.join(f'{attr}={self.__getattribute__(attr)}' for attr in self.__annotations__.keys())})"


@dataclass(frozen=True)
class IOAlergia(ParameterizedLearningAlgorithm):
    eps : float

    @property
    def params(self):
        ret = super().params
        ret.update(
            score_calc=ScoreFunctions.ScoreCalculation(ScoreFunctions.hoeffding_compatibility(self.eps)), eval_compat_on_pta=True, compatibility_behavior="future"
        )
        return ret


@dataclass(frozen=True)
class LikelihoodRatio(ParameterizedLearningAlgorithm):
    alpha : float

    @property
    def params(self):
        ret = super().params
        ret.update(
            score_calc=ScoreFunctions.ScoreCalculation(score_function=ScoreFunctions.likelihood_ratio_score(self.alpha))
        )
        return ret


@dataclass(frozen=True)
class AkaikeInformationCriterion(ParameterizedLearningAlgorithm):
    __annotations__ = dict()

    @property
    def params(self):
        ret = super().params
        ret.update(
            score_calc=ScoreFunctions.ScoreCalculation(score_function=ScoreFunctions.AIC_score())
        )
        return ret
