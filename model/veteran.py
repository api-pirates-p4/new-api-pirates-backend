"""
model/veteran.py

VeteranModel — ML model for PVO veteran eligibility prediction.
Internally refactored to follow SRP: each private method has ONE job.

Internal chain (mirrors SRP lesson's worker/orchestrator split):
    _build_training_data()   WORKER  — produces raw DataFrame
    _clean()                 WORKER  — encodes categoricals
    _train()                 WORKER  — fits LogReg + DecisionTree
    get_instance()           ORCHESTRATOR — coordinates the above three once

Public interface:
    predict(applicant)       WORKER  — returns prediction dict
    feature_weights()        WORKER  — returns importance dict
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import pandas as pd
import numpy as np


class VeteranModel:
    """
    Singleton ML model predicting whether a PVO application will be
    handled directly by PVO (outcome=1) or referred out (outcome=0).
    """

    _instance = None

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self):
        self.model    = None   # LogisticRegression  — probability output
        self.dt       = None   # DecisionTreeClassifier — feature importance
        self.encoder  = None   # OneHotEncoder for categorical columns
        self.features = None   # Final feature list after encoding
        self.data     = None   # Training DataFrame

        self._cat_cols = ['need_type', 'location', 'vet_status', 'employment']
        self._num_cols = ['housing_risk', 'household_sz', 'has_va_care']

    # =========================================================================
    # WORKERS — each does exactly ONE thing
    # =========================================================================

    def _build_training_data(self) -> pd.DataFrame:
        """
        WORKER: Combines hardcoded seed cases + real submissions from SQLite.
        Grows smarter as more people use the prescreener.
        """
        seed_records = [
            {'need_type':'rent',        'location':'poway',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':1,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'rent',        'location':'escondido',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':1,'household_sz':1,'has_va_care':1,'outcome':1},
            {'need_type':'rent',        'location':'poway',         'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':1,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'utility',     'location':'san_diego',     'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'utility',     'location':'escondido',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':1,'outcome':1},
            {'need_type':'utility',     'location':'poway',         'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':3,'has_va_care':0,'outcome':1},
            {'need_type':'utility',     'location':'santee',        'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'food',        'location':'poway',         'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'food',        'location':'santee',        'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'food',        'location':'spring_valley', 'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':1},
            {'need_type':'transport',   'location':'poway',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':1},
            {'need_type':'transport',   'location':'imperial_beach','vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'transport',   'location':'escondido',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'home_repair', 'location':'poway',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'home_repair', 'location':'ramona',        'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':1,'outcome':1},
            {'need_type':'home_repair', 'location':'vista',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':1},
            {'need_type':'home_repair', 'location':'san_marcos',    'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':1,'outcome':1},
            {'need_type':'home_repair', 'location':'santee',        'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'vehicle_reg', 'location':'poway',         'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'vehicle_reg', 'location':'vista',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'medical',     'location':'poway',         'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'medical',     'location':'san_diego',     'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'storage',     'location':'san_diego',     'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':1,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'moving',      'location':'san_diego',     'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'moving',      'location':'ramona',        'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':3,'has_va_care':0,'outcome':1},
            {'need_type':'food',        'location':'fallbrook',     'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'vehicle_reg', 'location':'fallbrook',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'home_repair', 'location':'carlsbad',      'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'home_repair', 'location':'pacific_beach', 'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':1},
            {'need_type':'rent',        'location':'mira_mesa',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':1},
            {'need_type':'utility',     'location':'chula_vista',   'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'food',        'location':'poway',         'vet_status':'dependent', 'employment':'employed',   'housing_risk':0,'household_sz':3,'has_va_care':0,'outcome':1},
            {'need_type':'home_repair', 'location':'poway',         'vet_status':'dependent', 'employment':'employed',   'housing_risk':0,'household_sz':2,'has_va_care':0,'outcome':1},
            {'need_type':'counseling',  'location':'poway',         'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':0},
            {'need_type':'counseling',  'location':'san_diego',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':0},
            {'need_type':'legal',       'location':'poway',         'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':0},
            {'need_type':'legal',       'location':'escondido',     'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':2,'has_va_care':1,'outcome':0},
            {'need_type':'mental_health','location':'san_diego',    'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':1,'household_sz':1,'has_va_care':0,'outcome':0},
            {'need_type':'mental_health','location':'poway',        'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':0},
            {'need_type':'medical',     'location':'outside_area',  'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':0},
            {'need_type':'counseling',  'location':'outside_area',  'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':0},
            {'need_type':'rent',        'location':'outside_area',  'vet_status':'veteran',   'employment':'unemployed', 'housing_risk':1,'household_sz':3,'has_va_care':0,'outcome':0},
            {'need_type':'legal',       'location':'outside_area',  'vet_status':'veteran',   'employment':'employed',   'housing_risk':0,'household_sz':1,'has_va_care':0,'outcome':0},
            {'need_type':'mental_health','location':'ramona',       'vet_status':'veteran',   'employment':'disabled',   'housing_risk':1,'household_sz':1,'has_va_care':1,'outcome':0},
            {'need_type':'counseling',  'location':'vista',         'vet_status':'veteran',   'employment':'disabled',   'housing_risk':0,'household_sz':1,'has_va_care':1,'outcome':0},
        ]
        seed_df = pd.DataFrame(seed_records)

        try:
            from database.db import get_connection
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT need_type, location, vet_status, employment,
                           housing_risk, household_sz, has_va_care,
                           CASE WHEN pvo_direct >= 0.5 THEN 1 ELSE 0 END AS outcome
                    FROM   prescreener_submissions
                    """
                ).fetchall()
            if rows:
                real_df  = pd.DataFrame([dict(r) for r in rows])
                df       = pd.concat([seed_df, real_df], ignore_index=True)
                print(f'[VeteranModel] Training on {len(seed_df)} seed + {len(real_df)} real = {len(df)} total rows')
            else:
                print(f'[VeteranModel] No real submissions yet — using {len(seed_df)} seed rows only')
                df = seed_df
        except Exception as e:
            print(f'[VeteranModel] Could not load real submissions ({e}) — using seed data only')
            df = seed_df

        majority  = df[df['outcome'] == 1]
        minority  = df[df['outcome'] == 0]
        if len(minority) > 0 and len(majority) > 0:
            upsampled = resample(minority, replace=True,
                                 n_samples=len(majority), random_state=42)
            return pd.concat([majority, upsampled]).sample(
                frac=1, random_state=42).reset_index(drop=True)
        return df

    def _clean(self) -> None:
        """
        WORKER: One-hot encodes categorical columns in self.data in-place.
        Does NOT train models or build data.
        """
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded  = self.encoder.fit_transform(self.data[self._cat_cols])
        enc_cols = self.encoder.get_feature_names_out(self._cat_cols).tolist()
        enc_df   = pd.DataFrame(encoded, columns=enc_cols, index=self.data.index)
        self.data    = pd.concat([self.data[self._num_cols + ['outcome']], enc_df], axis=1)
        self.features = self._num_cols + enc_cols

    def _train(self) -> None:
        """
        WORKER: Fits LogisticRegression and DecisionTreeClassifier on self.data.
        Prints accuracy. Does NOT build data or encode.
        """
        X = self.data[self.features]
        y = self.data['outcome']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)

        self.dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.dt.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f'[VeteranModel] accuracy: {acc:.2%}')

    def _encode_applicant(self, applicant: dict) -> pd.DataFrame:
        """
        WORKER: Encodes a single applicant dict into a feature-aligned DataFrame.
        Does NOT run the model or touch the database.
        """
        row = pd.DataFrame([{
            'need_type':    applicant['need_type'],
            'location':     applicant['location'],
            'vet_status':   applicant['vet_status'],
            'employment':   applicant['employment'],
            'housing_risk': int(applicant.get('housing_risk', 0)),
            'household_sz': int(applicant.get('household_sz', 1)),
            'has_va_care':  int(applicant.get('has_va_care', 0)),
        }])
        encoded  = self.encoder.transform(row[self._cat_cols])
        enc_cols = self.encoder.get_feature_names_out(self._cat_cols).tolist()
        enc_df   = pd.DataFrame(encoded, columns=enc_cols)
        return pd.concat([row[self._num_cols].reset_index(drop=True), enc_df], axis=1)

    def _compute_confidence(self, direct_proba: float, refer_proba: float) -> str:
        """
        WORKER: Maps probability scores to a confidence label.
        Pure function — no side effects.
        """
        top = max(direct_proba, refer_proba)
        if top >= 0.75: return 'high'
        if top >= 0.60: return 'medium'
        return 'low'

    def _get_top_factors(self, n: int = 3) -> list:
        """
        WORKER: Returns top-N feature names by decision tree importance.
        """
        pairs = sorted(zip(self.features, self.dt.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        return [f for f, _ in pairs[:n]]

    # =========================================================================
    # ORCHESTRATOR — coordinates the singleton build
    # =========================================================================

    @classmethod
    def get_instance(cls) -> 'VeteranModel':
        """
        ORCHESTRATOR: Builds the singleton by chaining the three workers:
            _build_training_data → _clean → _train

        Subsequent calls return the cached instance immediately.
        """
        if cls._instance is None:
            instance       = cls()
            instance.data  = instance._build_training_data()   # Step 1
            instance._clean()                                   # Step 2
            instance._train()                                   # Step 3
            cls._instance  = instance
        return cls._instance

    # =========================================================================
    # PUBLIC WORKERS — called by service/prediction_service.py
    # =========================================================================

    def predict(self, applicant: dict) -> dict:
        """
        WORKER: Runs inference for one applicant.
        Input is assumed already validated by service/validator.py.

        Returns:
            { pvo_direct, refer_out, confidence, top_factors }
        """
        X = self._encode_applicant(applicant)
        refer_proba, direct_proba = np.squeeze(self.model.predict_proba(X))
        return {
            'pvo_direct':  round(float(direct_proba), 4),
            'refer_out':   round(float(refer_proba),  4),
            'confidence':  self._compute_confidence(direct_proba, refer_proba),
            'top_factors': self._get_top_factors(),
        }

    def feature_weights(self) -> dict:
        """
        WORKER: Returns {feature_name: importance_score} for all non-zero features.
        """
        return {f: round(float(i), 4)
                for f, i in zip(self.features, self.dt.feature_importances_)
                if i > 0}


# ── Init / test helpers ───────────────────────────────────────────────────────

def initVeteran() -> None:
    """Warms the singleton at app startup. Call from generate_data CLI command."""
    VeteranModel.get_instance()


def testVeteran() -> None:
    """Smoke-tests predict() with two representative cases."""
    model = VeteranModel.get_instance()

    cases = [
        {'label': 'Poway veteran — utility, disabled',
         'data':  {'need_type':'utility','location':'poway','vet_status':'veteran',
                   'employment':'disabled','housing_risk':0,'household_sz':2,'has_va_care':1}},
        {'label': 'Out-of-area veteran — legal, unemployed',
         'data':  {'need_type':'legal','location':'outside_area','vet_status':'veteran',
                   'employment':'unemployed','housing_risk':0,'household_sz':1,'has_va_care':0}},
    ]

    for case in cases:
        r = model.predict(case['data'])
        print(f"\n{case['label']}")
        print(f"  PVO direct: {r['pvo_direct']:.2%}  |  Refer out: {r['refer_out']:.2%}")
        print(f"  Confidence: {r['confidence']}  |  Top factors: {r['top_factors']}")

    print("\nTop feature weights:")
    for feat, imp in sorted(model.feature_weights().items(),
                            key=lambda x: x[1], reverse=True)[:6]:
        print(f"  {feat}: {imp:.2%}")


if __name__ == '__main__':
    testVeteran()
