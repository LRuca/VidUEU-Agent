import pytest

from orchestrator.iteration.feedback import validate_feedback


def test_held_out_metrics_cannot_enter_control_feedback():
    with pytest.raises(ValueError):
        validate_feedback(
            {
                "feedback_metrics": {
                    "MMDU": 0.3,
                    "GEdit-Bench-En": 6.0,
                    "WEAVEbench": 0.4,
                    "MMBench": 0.5,
                },
                "held_out_metrics": {"MMBench": None, "MMStar": None, "VSI-Bench": None},
            }
        )
