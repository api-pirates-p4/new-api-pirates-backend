"""
service/validator.py
RESPONSIBILITY: One thing only — validate and sanitize applicant input dicts.
"""

VALIDATION_ERRORS = {
    'MISSING_FIELDS':    'MISSING_FIELDS',
    'INVALID_NEED_TYPE': 'INVALID_NEED_TYPE',
    'INVALID_LOCATION':  'INVALID_LOCATION',
    'INVALID_VET_STATUS':'INVALID_VET_STATUS',
    'INVALID_EMPLOYMENT':'INVALID_EMPLOYMENT',
    'INVALID_NUMERIC':   'INVALID_NUMERIC',
}

VALID_NEED_TYPES = {
    'rent', 'utility', 'food', 'transport', 'home_repair',
    'vehicle_reg', 'medical', 'storage', 'moving',
    'counseling', 'legal', 'mental_health',
}

VALID_LOCATIONS = {
    'poway', 'ramona', 'escondido', 'vista', 'santee', 'san_marcos',
    'spring_valley', 'mira_mesa', 'carlsbad', 'imperial_beach',
    'pacific_beach', 'chula_vista', 'fallbrook', 'san_diego', 'outside_area',
}

VALID_VET_STATUSES  = {'veteran', 'dependent'}
VALID_EMPLOYMENTS   = {'employed', 'unemployed', 'disabled'}
REQUIRED_FIELDS     = ['need_type', 'location', 'vet_status', 'employment']


class ValidationError(Exception):
    def __init__(self, error_type: str, detail: str = ''):
        self.error_type = error_type
        self.detail     = detail
        super().__init__(f"{error_type}: {detail}")


def check_required_fields(data: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] is None]
    if missing:
        raise ValidationError(VALIDATION_ERRORS['MISSING_FIELDS'], f"Missing: {missing}")


def validate_need_type(value: str) -> str:
    if value not in VALID_NEED_TYPES:
        raise ValidationError(VALIDATION_ERRORS['INVALID_NEED_TYPE'], f"'{value}' not allowed")
    return value


def validate_location(value: str) -> str:
    if value not in VALID_LOCATIONS:
        raise ValidationError(VALIDATION_ERRORS['INVALID_LOCATION'], f"'{value}' not allowed")
    return value


def validate_vet_status(value: str) -> str:
    if value not in VALID_VET_STATUSES:
        raise ValidationError(VALIDATION_ERRORS['INVALID_VET_STATUS'], f"'{value}' not allowed")
    return value


def validate_employment(value: str) -> str:
    if value not in VALID_EMPLOYMENTS:
        raise ValidationError(VALIDATION_ERRORS['INVALID_EMPLOYMENT'], f"'{value}' not allowed")
    return value


def coerce_numeric_fields(data: dict) -> dict:
    try:
        return {
            **data,
            'housing_risk': int(data.get('housing_risk', 0)),
            'household_sz': max(1, int(data.get('household_sz', 1))),
            'has_va_care':  int(data.get('has_va_care', 0)),
        }
    except (TypeError, ValueError) as exc:
        raise ValidationError(VALIDATION_ERRORS['INVALID_NUMERIC'], str(exc)) from exc


def validate_applicant(data: dict) -> dict:
    """ORCHESTRATOR: runs all validation workers in sequence."""
    check_required_fields(data)
    validate_need_type(data['need_type'])
    validate_location(data['location'])
    validate_vet_status(data['vet_status'])
    validate_employment(data['employment'])
    return coerce_numeric_fields(data)