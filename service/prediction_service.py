"""
service/prediction_service.py
RESPONSIBILITY: One thing only — call VeteranModel and return a prediction dict.
"""

from model.veteran import VeteranModel


class PredictionError(Exception):
    pass


def run_prediction(applicant: dict) -> dict:
    try:
        model = VeteranModel.get_instance()
        return model.predict(applicant)
    except Exception as exc:
        raise PredictionError(f"ML prediction failed: {exc}") from exc


def get_feature_weights() -> dict:
    try:
        model = VeteranModel.get_instance()
        return model.feature_weights()
    except Exception as exc:
        raise PredictionError(f"Feature weights unavailable: {exc}") from exc