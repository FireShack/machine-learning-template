import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class StandarizationScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, y=None):
        return self
    def transform(self, x_set, y=None):
        x_set_copy = x_set.copy()
        robust_scaler = RobustScaler()
        x_scaled = robust_scaler.fit_transform(x_set_copy[self.attributes])
        x_scaled = pd.DataFrame(x_scaled, columns=self.attributes, index=x_set_copy.index)
        for attr in self.attributes:
            x_set_copy[attr] = x_scaled[attr]
        return x_set_copy
    