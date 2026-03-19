"""
service/response_formatter.py
RESPONSIBILITY: One thing only — shape internal dicts into HTTP response bodies.
"""

from service.validator import ValidationError, VALIDATION_ERRORS
from service.prediction_service import PredictionError


ERROR_MESSAGES = {
    VALIDATION_ERRORS['MISSING_FIELDS']:     'Missing required fields in your submission.',
    VALIDATION_ERRORS['INVALID_NEED_TYPE']:  'Unrecognised need type.',
    VALIDATION_ERRORS['INVALID_LOCATION']:   'Unrecognised location.',
    VALIDATION_ERRORS['INVALID_VET_STATUS']: 'Veteran status must be "veteran" or "dependent".',
    VALIDATION_ERRORS['INVALID_EMPLOYMENT']: 'Employment must be "employed", "unemployed", or "disabled".',
    VALIDATION_ERRORS['INVALID_NUMERIC']:    'Numeric fields are invalid.',
    'PREDICTION_ERROR':                      'ML model could not produce a prediction. Please try again.',
    'NOT_FOUND':                             'The requested resource was not found.',
    'DEFAULT':                               'An unexpected error occurred.',
}


def format_prediction_response(prediction: dict, submission_id: int) -> dict:
    return {
        'submission_id': submission_id,
        'pvo_direct':    prediction['pvo_direct'],
        'refer_out':     prediction['refer_out'],
        'confidence':    prediction['confidence'],
        'top_factors':   prediction['top_factors'],
    }


def format_error_response(exc: Exception) -> tuple:
    if isinstance(exc, ValidationError):
        message = ERROR_MESSAGES.get(exc.error_type, ERROR_MESSAGES['DEFAULT'])
        return {'error': message, 'detail': exc.detail}, 400

    if isinstance(exc, PredictionError):
        return {'error': ERROR_MESSAGES['PREDICTION_ERROR'], 'detail': str(exc)}, 500

    return {'error': ERROR_MESSAGES['DEFAULT'], 'detail': str(exc)}, 500


def format_stats_response(stats: dict) -> dict:
    return stats


def format_submission_response(submission: dict) -> dict:
    return submission