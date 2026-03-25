"""
api/veteran_api.py

ORCHESTRATORS ONLY — this file contains zero business logic.

Each endpoint method is an orchestrator that:
  1. Delegates validation  → service/validator.py
  2. Delegates prediction  → service/prediction_service.py
  3. Delegates persistence → database/submission_repository.py
  4. Delegates formatting  → service/response_formatter.py

Mirrors the SRP lesson's loadUserGroups() orchestrator pattern:
    validate → predict → save → format → respond

Registration in main.py:
    from api.veteran_api import veteran_api
    from database.schema import create_tables
    from model.veteran import initVeteran

    app.register_blueprint(veteran_api)

    @custom_cli.command('generate_data')
    def generate_data():
        create_tables()
        initVeteran()
"""

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

# ── Single-responsibility workers imported from their own modules ─────────────
from service.validator          import validate_applicant, ValidationError
from service.prediction_service import run_prediction, get_feature_weights, PredictionError
from service.response_formatter import (
    format_prediction_response,
    format_error_response,
    format_stats_response,
    format_submission_response,
)
from database.submission_repository import (
    save_submission,
    get_submission_by_id,
    get_recent_submissions,
    get_submission_stats,
    SubmissionNotFound,
    SubmissionWriteError,
)
from database.importance_repository import (
    log_feature_importances,
    get_latest_feature_importances,
)

# ── Blueprint setup ───────────────────────────────────────────────────────────
veteran_api = Blueprint('veteran_api', __name__, url_prefix='/api/veteran')
api = Api(veteran_api)


# ── Error type constants (mirrors ERROR_TYPES from SRP lesson) ────────────────
API_ERRORS = {
    'NO_BODY':       'NO_REQUEST_BODY',
    'NOT_FOUND':     'SUBMISSION_NOT_FOUND',
    'WRITE_FAILED':  'SUBMISSION_WRITE_FAILED',
}


class VeteranAPI:

    # ── POST /api/veteran/predict ─────────────────────────────────────────────
    class _Predict(Resource):
        def post(self):
            """
            ORCHESTRATOR: Accept quiz answers → validate → predict → save → respond.

            Chain (mirrors SRP lesson's async/await pattern):
                parse body
                    → validate_applicant()        [validator worker]
                    → run_prediction()            [prediction worker]
                    → save_submission()           [repository worker]
                    → format_prediction_response()[formatter worker]
                    → jsonify + return

            One .catch() equivalent: the try/except at the end.

            Request body:
                {
                  "need_type":    "utility",
                  "location":     "poway",
                  "vet_status":   "veteran",
                  "employment":   "disabled",
                  "housing_risk": 0,
                  "household_sz": 2,
                  "has_va_care":  1
                }

            Response 200:
                {
                  "submission_id": 42,
                  "pvo_direct":    0.83,
                  "refer_out":     0.17,
                  "confidence":    "high",
                  "top_factors":   ["need_type_utility", ...]
                }
            """
            # ── Step 0: Parse body ────────────────────────────────────────────
            body = request.get_json()
            if not body:
                return {'error': API_ERRORS['NO_BODY']}, 400

            try:
                # ── Step 1: Validate ──────────────────────────────────────────
                clean_answers = validate_applicant(body)

                # ── Step 2: Predict ───────────────────────────────────────────
                prediction = run_prediction(clean_answers)

                # ── Step 3: Persist ───────────────────────────────────────────
                submission_id = save_submission(clean_answers, prediction)

                # ── Step 4: Format + respond ──────────────────────────────────
                response_body = format_prediction_response(prediction, submission_id)
                return jsonify(response_body)

            except (ValidationError, PredictionError, SubmissionWriteError) as exc:
                # ── Centralised error handler (mirrors .catch() in SRP lesson) ─
                body, status = format_error_response(exc)
                return body, status

    # ── GET /api/veteran/submission/<id> ──────────────────────────────────────
    class _GetSubmission(Resource):
        def get(self, submission_id: int):
            """
            ORCHESTRATOR: Fetch a saved submission by id.

            Chain:
                get_submission_by_id()    [repository worker]
                    → format_submission_response() [formatter worker]
                    → jsonify + return
            """
            try:
                submission = get_submission_by_id(submission_id)
                return jsonify(format_submission_response(submission))
            except SubmissionNotFound:
                return {'error': API_ERRORS['NOT_FOUND'],
                        'detail': f'submission id={submission_id}'}, 404

    # ── GET /api/veteran/submissions ──────────────────────────────────────────
    class _ListSubmissions(Resource):
        def get(self):
            """
            ORCHESTRATOR: Return the 50 most recent submissions.

            Chain:
                get_recent_submissions()  [repository worker]
                    → format each         [formatter worker]
                    → jsonify + return
            """
            limit = min(int(request.args.get('limit', 50)), 200)
            rows  = get_recent_submissions(limit)
            return jsonify([format_submission_response(r) for r in rows])

    # ── GET /api/veteran/stats ────────────────────────────────────────────────
    class _Stats(Resource):
        def get(self):
            """
            ORCHESTRATOR: Aggregate stats across all submissions.
            Used by admin dashboard.

            Chain:
                get_submission_stats()    [repository worker]
                    → format_stats_response() [formatter worker]
                    → jsonify + return
            """
            stats = get_submission_stats()
            return jsonify(format_stats_response(stats))

    # ── GET /api/veteran/weights ──────────────────────────────────────────────
    class _FeatureWeights(Resource):
        def get(self):
            """
            ORCHESTRATOR: Return feature importance scores.
            Tries DB cache first; falls back to live model.

            Chain:
                get_latest_feature_importances()  [repository worker]
                    → if empty: get_feature_weights() + log_feature_importances()
                    → jsonify + return
            """
            cached = get_latest_feature_importances()

            if cached:
                return jsonify(cached)

            # No cached snapshot — compute from live model and cache it
            try:
                weights = get_feature_weights()
                log_feature_importances(weights)
                return jsonify([
                    {'feature_name': k, 'importance': v}
                    for k, v in sorted(weights.items(),
                                       key=lambda x: x[1], reverse=True)
                ])
            except PredictionError as exc:
                body, status = format_error_response(exc)
                return body, status

    # ── Register routes ───────────────────────────────────────────────────────
    api.add_resource(_Predict,          '/predict')
    api.add_resource(_GetSubmission,    '/submission/<int:submission_id>')
    api.add_resource(_ListSubmissions,  '/submissions')
    api.add_resource(_Stats,            '/stats')
    api.add_resource(_FeatureWeights,   '/weights')
    class _Retrain(Resource):
        def post(self):
            """
            ORCHESTRATOR: POST /api/veteran/retrain
            Clears the singleton and retrains on seed + all saved submissions.
            """
            try:
                from model.veteran import VeteranModel, initVeteran
                VeteranModel._instance = None
                initVeteran()
                weights = get_feature_weights()
                log_feature_importances(weights)
                return jsonify({'status': 'retrained', 'message': 'Model retrained on seed + real submissions'})
            except Exception as exc:
                body, status = format_error_response(exc)
                return body, status

    api.add_resource(_Retrain, '/retrain')