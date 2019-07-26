"""BasePipeline Class."""
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.pipeline import Columns

# Numerical features used for analysis.
NUMERICAL_FEATURES = [
    'speed',
    'car_accel',
    'lateral_acceleration',
    'rpm',
    'pitch',
    'shift'
]

# Categorical features used for analysis.
CATEGORICAL_FEATURES = [
    'visibility',
    'rain_intensity',
    'driver_wellbeing'
]


class BasePipeline():
    """Add first features preprocessing step to Pipeline."""

    def __init__(self, model):
        """Initialise a sklearn Pipeline instance.

        Parameters
        ----------
        model : object
            The model to be used by the pipeline.

        """
        features = self.build_features_step()
        self.custom_steps = [
            ('features', features),
            ('mdl', model)
        ]

    def make_pipeline(self, steps=None):
        """Construct a Pipeline from the given built-in steps + given steps.

        Parameters
        ----------
        steps : list
            List on estimators.

        Returns
        -------
        Pipeline
            Pipeline including built-in features preprocessing steps and model.

        """
        pipe_steps = self.custom_steps + steps if steps else self.custom_steps

        return Pipeline(pipe_steps)

    def build_features_step(self):
        """Build the features preprocessing Pipeline steps.

        Returns
        -------
        FeatureUnion
            Features preprocessing Pipeline steps.

        """
        numeric = self.get_numerical_metric_feature_names()

        # Build feature engineering pipeline step
        features = FeatureUnion([
                ('numeric',
                 make_pipeline(Columns(names=numeric), StandardScaler())),
                ('categorical',
                 make_pipeline(
                     Columns(names=CATEGORICAL_FEATURES),
                     OneHotEncoder(sparse=False)))
            ])

        return features

    def get_numerical_metric_feature_names(self):
        """Return list of metrics features names.

        Returns
        -------
        list
            List of metrics features names.

        """
        # metrics dataset column suffixes
        col_name_suffix = [
            'mean',
            'std',
            'min',
            '25',
            '50',
            '75',
            'max'
        ]

        # Build metrics dataset columns names.
        num_feat_col_names = [feature + '_' + suffix
                              for feature in NUMERICAL_FEATURES
                              for suffix in col_name_suffix]

        return num_feat_col_names
