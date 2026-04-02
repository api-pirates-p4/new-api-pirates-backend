# titanic.py
# Titanic Model — trains on the Titanic dataset and predicts passenger survival.
# Follows Model/View/Control pattern: this file is the Model.

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns


class TitanicModel:
    """A class used to represent the Titanic Model for passenger survival prediction."""

    # Singleton instance — model is trained once and reused for all predictions
    _instance = None

    def __init__(self):
        # The trained ML models
        self.model = None   # Logistic Regression (primary prediction model)
        self.dt = None      # Decision Tree (used for feature importance)

        # Define ML features and target
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'

        # Load the Titanic dataset via seaborn
        self.titanic_data = sns.load_dataset('titanic')

        # One-hot encoder for the 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        """Clean the Titanic dataset and prepare it for training.

        Steps:
        - Drop unnecessary columns
        - Convert boolean/categorical columns to numeric
        - One-hot encode 'embarked'
        - Drop rows with missing values
        """
        # Drop columns not used in the model
        self.titanic_data.drop(
            ['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'],
            axis=1, inplace=True
        )

        # Convert sex to binary: male=1, female=0
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(
            lambda x: 1 if x == 'male' else 0
        )

        # Convert alone to binary: True=1, False=0
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(
            lambda x: 1 if x == True else 0
        )

        # Drop rows missing 'embarked' before encoding
        self.titanic_data.dropna(subset=['embarked'], inplace=True)

        # One-hot encode 'embarked' column (C, Q, S)
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.titanic_data.drop(['embarked'], axis=1, inplace=True)

        # Add one-hot encoded columns to features list
        self.features.extend(cols)

        # Final drop of any remaining rows with missing values
        self.titanic_data.dropna(inplace=True)

    def _train(self):
        """Train the Logistic Regression and Decision Tree models.

        - LogisticRegression is the primary model used for survival prediction.
        - DecisionTreeClassifier is used to expose feature importances.
        """
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]

        # Train Logistic Regression (primary model)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        # Train Decision Tree (for feature importance)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls):
        """Get (or create) the singleton TitanicModel instance.

        Cleans and trains the model on first call. Returns the cached
        instance on subsequent calls — so training only happens once.

        Returns:
            TitanicModel: the singleton instance, ready for prediction.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, passenger):
        """Predict the survival probability of a passenger.

        Args:
            passenger (dict): A dictionary with the following keys:
                'pclass'   : int   — Passenger class (1, 2, or 3)
                'sex'      : str   — 'male' or 'female'
                'age'      : float — Age in years
                'sibsp'    : int   — Number of siblings/spouses aboard
                'parch'    : int   — Number of parents/children aboard
                'fare'     : float — Ticket fare (0–512)
                'embarked' : str   — Port of embarkation ('C', 'Q', or 'S')
                'alone'    : bool  — True if traveling alone

        Returns:
            dict: { 'die': float, 'survive': float } — probabilities summing to 1.0
        """
        # Build a DataFrame — wrap scalar values in lists if needed
        # (frontend JSON sends scalars; testTitanic sends lists)
        passenger = {k: v if isinstance(v, list) else [v] for k, v in passenger.items()}
        passenger_df = pd.DataFrame(passenger, index=[0])

        # Encode sex and alone to binary
        passenger_df['sex'] = passenger_df['sex'].apply(
            lambda x: 1 if x == 'male' else 0
        )
        passenger_df['alone'] = passenger_df['alone'].apply(
            lambda x: 1 if x == True else 0
        )

        # One-hot encode 'embarked'
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)

        # Drop columns not used in prediction
        passenger_df.drop(['embarked'], axis=1, inplace=True)
        if 'name' in passenger_df.columns:
            passenger_df.drop(['name'], axis=1, inplace=True)

        # Predict survival probability — index 0=die, 1=survive
        proba = self.model.predict_proba(passenger_df)[0]
        die, survive = proba[0], proba[1]
        return {'die': die, 'survive': survive}

    def feature_weights(self):
        """Get the relative importance of each feature from the Decision Tree.

        Returns:
            dict: { feature_name: importance_score } for each feature.
                  Scores are floats between 0 and 1, summing to 1.0.
        """
        importances = self.dt.feature_importances_
        return {
            feature: importance
            for feature, importance in zip(self.features, importances)
        }


# ── Initialization helper (called from generate_data CLI command) ─────────────

def initTitanic():
    """Initialize the Titanic Model at app startup.

    Loads and trains the model into memory so that the first prediction
    request is not delayed by training time.

    """
    TitanicModel.get_instance()


# ── Test / development helper ─────────────────────────────────────────────────

def testTitanic():
    """Test the TitanicModel with a sample passenger (John Mortensen).

    Prints:
        - Passenger input data
        - Survival and death probabilities
        - Feature importance weights
    """
    print("\n Step 1: Define theoretical passenger data for prediction:")
    passenger = {
        'pclass':    [2],
        'sex':       ['male'],
        'age':       [66],
        'sibsp':     [1],
        'parch':     [1],
        'fare':      [16.00],
        'embarked':  ['S'],
        'alone':     [False]
    }
    print("\t", passenger)
    print()

    # Get the singleton instance (cleans + trains on first call)
    titanicModel = TitanicModel.get_instance()
    print(" Step 2:", titanicModel.get_instance.__doc__)

    # Predict survival probability
    print(" Step 3:", titanicModel.predict.__doc__)
    probability = titanicModel.predict(passenger)
    print('\t Death probability:    {:.2%}'.format(probability.get('die')))
    print('\t Survival probability: {:.2%}'.format(probability.get('survive')))
    print()

    # Print feature importances
    print(" Step 4:", titanicModel.feature_weights.__doc__)
    importances = titanicModel.feature_weights()
    for feature, importance in importances.items():
        print(f"\t\t {feature:<20} {importance:.2%}")


if __name__ == "__main__":
    print(" Begin:", testTitanic.__doc__)
    testTitanic()