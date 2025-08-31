from typing import TypedDict, List

class MetricsDict(TypedDict):
    roc_auc: float
    pr_auc: float

FeatureColumns = List[str]