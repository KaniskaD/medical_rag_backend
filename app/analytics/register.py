from app.analytics.registry import register_analytics
from app.analytics.modules import (
    lab_analytics,
    image_analytics,
    audio_analytics,
)

register_analytics(
    name="lab",
    requires=["lab"],
    func=lab_analytics,
)

register_analytics(
    name="image",
    requires=["image"],
    func=image_analytics,
)

register_analytics(
    name="audio",
    requires=["audio"],
    func=audio_analytics,
)
