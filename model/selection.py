
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
class AggregateModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, ratios=[]):
        self.models_ = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for m in self.models_:
            m.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            m.predict(X) for m in self.models_
        ])
        # return np.mean(predictions, axis=1)
        # print(predictions.shape)
        # print(self.ratios)
        # print(predictions)
        # return np.matrix(predictions) * np.transpose(np.matrix(self.ratios))
        return np.sum(predictions * self.ratios, axis=1)

    def validate(self, X, y):
        # make predictions
        predictions = self.predict(X)
        # print(predictions)
        # print(predictions.shape)
        # print(y.shape)
        return np.sqrt(mean_squared_error(predictions, y))

    def get_valid_weights(self, r, l):
        self.valid_ratios = []
        self.get_valid_weights_(0, r, l - 1, [])

    def get_valid_weights_(self, curr_depth, r, l, c):
        if curr_depth == l:
            if sum(c) <= 1:
                c.append(1 - sum(c))
                # print(c)
                self.valid_ratios.append(
                    c.copy())  # very important, otherwise, a pointer to c is appended which changes by the stack operation
                c.pop()
                # print(self.valid_ratios)
            return
        else:
            # print(a[curr_idx])
            # c.append(a[curr_idx])
            for child_idx in range(len(r)):
                c.append(r[child_idx])
                self.get_valid_weights_(curr_depth + 1, r, l, c)
                c.pop()
                # curr_depth += 1

    def get_best_ratios(self, X_val, y_val):
        res = 0.1
        l = 10
        r = np.multiply(range(l + 1), res)
        # print(r)
        # print(len(self.models))
        self.get_valid_weights(r, len(self.models_))
        # print(self.valid_ratios)
        best_rmse = 10000

        best_ratios = self.valid_ratios[0]

        for ratios in self.valid_ratios:
            # self.ratios = np.array(ratios)
            self.ratios = ratios
            # print(self.ratios.shape)
            # print(self.ratios)
            rmse = self.validate(X_val, y_val)
            # print(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_ratios = ratios
        self.ratios = best_ratios
        return best_ratios, best_rmse