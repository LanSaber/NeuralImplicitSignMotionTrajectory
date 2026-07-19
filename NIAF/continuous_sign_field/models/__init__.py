from NIAF.continuous_sign_field.models.local_implicit import LocalAmortizedImplicitResidualField
from NIAF.continuous_sign_field.models.meta_implicit import MetaImplicitResidualField
from NIAF.continuous_sign_field.models.residual_flow_transformer import ResidualFlowTransformer

__all__ = [
    "LocalAmortizedImplicitResidualField",
    "MetaImplicitResidualField",
    "ResidualFlowTransformer",
    "RetrievalConfidenceAdaptiveField",
]


def __getattr__(name):
    if name == "RetrievalConfidenceAdaptiveField":
        from NIAF.retrieval_confidence_field.models import RetrievalConfidenceAdaptiveField

        return RetrievalConfidenceAdaptiveField
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
