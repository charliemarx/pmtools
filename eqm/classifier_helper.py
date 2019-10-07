import numpy as np
from copy import deepcopy
from inspect import getfullargspec

def get_classifier_from_coefficients(coefs):
    return ClassificationModel(predict_handle=lambda X: 1,
                               model_info={'coefficients': coefs[1:], 'intercept': coefs[0]},
                               model_type=ClassificationModel.LINEAR_MODEL_TYPE)


class ClassificationModel(object):

    LINEAR_MODEL_TYPE = 'linear'
    SUPPORTED_MODEL_TYPES = [LINEAR_MODEL_TYPE]

    def __init__(self, predict_handle, model_type, model_info, training_info = None):

        if training_info is None:
            training_info = dict()

        # check predict handle
        assert callable(predict_handle)
        spec = getfullargspec(predict_handle)
        assert 'X' in spec.args

        # check other fields
        assert isinstance(model_type, str)
        assert isinstance(model_info, dict)
        assert isinstance(training_info, dict)
        assert model_type in ClassificationModel.SUPPORTED_MODEL_TYPES, "unsupported model type"

        # initialize properties
        self.predict_handle = predict_handle
        self._model_type = str(model_type)
        self._model_info = deepcopy(model_info)
        self._training_info = deepcopy(training_info)

        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            self._coefficients = np.array(self._model_info['coefficients'])
            self._intercept = np.array(self._model_info['intercept'])
            self.predict_handle = lambda X: np.sign(X[:, 1:].dot(self._coefficients) + self._intercept)
            self.score_handle = lambda X: X[:, 1:].dot(self._coefficients) + self._intercept
        else:
            self.score_handle = self.predict_handle

        assert self.check_rep()


    def predict(self, X):
        return np.array(self.predict_handle(X)).flatten()

    def score(self, X):
        return np.array(self.score_handle(X)).flatten()

    @property
    def model_type(self):
        return str(self._model_type)


    @property
    def model_info(self):
        return deepcopy(self._model_info)

    @property
    def coefficients(self):
        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            return np.array(self._model_info['coefficients']).flatten()

    @property
    def intercept(self):
        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            return np.array(self._model_info['intercept'])

    @property
    def training_info(self):
        return deepcopy(self._training_info)


    def check_rep(self):
        assert callable(self.predict_handle)
        assert isinstance(self.model_type, str)
        assert isinstance(self.model_info, dict)
        assert isinstance(self.training_info, dict)
        return True
