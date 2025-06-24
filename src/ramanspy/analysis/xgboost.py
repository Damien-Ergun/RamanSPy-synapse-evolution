import xgboost as xgb

from .Step import AnalysisStep


class XGBoost(AnalysisStep):
    """Train and apply an XGBoost classifier.

    Parameters
    ----------
    train_dataset : tuple of array_like
        Training data and labels as ``(X_train, y_train)``.
    test_dataset : tuple of array_like, optional
        Optional validation data as ``(X_test, y_test)``.
    **kwargs :
        Hyperparameters passed to :class:`xgboost.XGBClassifier`.
    """

    def __init__(self, *, train_dataset, test_dataset=None, **kwargs):
        super().__init__(_xgboost, train_dataset, test_dataset, **kwargs)


def _xgboost(data_to_predict, train_dataset, test_dataset=None, **kwargs):
    X_train, y_train = train_dataset

    model = xgb.XGBClassifier(**kwargs)
    if test_dataset is not None:
        X_test, y_test = test_dataset
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(data_to_predict)
    preds = preds.reshape(-1, 1)

    return preds, model
