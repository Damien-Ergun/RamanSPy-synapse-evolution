import xgboost as xgb
from typing import Optional, Tuple

class XGBoost:
    """Wrapper around :class:`xgboost.XGBClassifier`.

    Parameters
    ----------
    train_data : Tuple[array_like, array_like]
        Training features and labels.
    test_data : Tuple[array_like, array_like], optional
        Optional test features and labels for evaluation.
    **params : Any
        Hyper-parameters passed to ``xgboost.XGBClassifier``.
    """

    def __init__(self, train_data: Tuple, test_data: Optional[Tuple] = None, **params):
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.model: Optional[xgb.XGBClassifier] = None

    def apply(self):
        """Train an ``XGBClassifier`` on the provided data and return the fitted model."""
        X_train, y_train = self.train_data
        eval_set = None
        if self.test_data is not None:
            X_test, y_test = self.test_data
            eval_set = [(X_test, y_test)]

        model = xgb.XGBClassifier(**self.params)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.model = model
        return model
