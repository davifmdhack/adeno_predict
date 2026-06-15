from typing import TypedDict


class MetricsDict(TypedDict):
    roc_auc: float
    pr_auc: float


FeatureColumns = list[str]
