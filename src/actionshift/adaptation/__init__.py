"""End-to-end adaptation adapters wired to frozen canonical task backbones."""

from actionshift.adaptation.adapters import (
    ContractAdapter,
    ExactBeliefAdapter,
    NoAdaptAdapter,
    OracleAdapter,
)
from actionshift.adaptation.dualabi_adapter import DualABIProbeAdapter
from actionshift.adaptation.factorized_grammar import (
    FactorizedGrammarAdapter,
    FactorizedGrammarDriver,
    FactorizedGrammarProbingAdapter,
)
from actionshift.adaptation.hypotheses import ExactBeliefDriver, HypothesisSimulator
from actionshift.adaptation.probe_osi import ProbeOsiAdapter
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiAdapter,
    RecurrentOsiRegressor,
    RunningLagFeatures,
)
from actionshift.adaptation.response import ResponseModel
from actionshift.adaptation.scale_corrector import ScaleCorrector

__all__ = [
    "ContractAdapter",
    "DualABIProbeAdapter",
    "ExactBeliefAdapter",
    "ExactBeliefDriver",
    "FactorizedGrammarAdapter",
    "FactorizedGrammarDriver",
    "FactorizedGrammarProbingAdapter",
    "HypothesisSimulator",
    "NoAdaptAdapter",
    "OracleAdapter",
    "ProbeOsiAdapter",
    "RecurrentOsiAdapter",
    "RecurrentOsiRegressor",
    "ResponseModel",
    "RunningLagFeatures",
    "ScaleCorrector",
]
