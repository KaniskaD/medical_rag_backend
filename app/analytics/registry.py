from typing import Dict, Callable, List

AnalyticsResult = Dict
AnalyticsFunc = Callable[[Dict], AnalyticsResult]

ANALYTICS_REGISTRY: Dict[str, Dict] = {}

def register_analytics(
    name: str,
    requires: List[str],
    func: AnalyticsFunc,
):
    ANALYTICS_REGISTRY[name] = {
        "requires": requires,
        "func": func,
    }
