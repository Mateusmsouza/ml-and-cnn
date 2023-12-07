import math
from statistics import mean, stdev
from sklearn.metrics import r2_score


class LinearRegression:


    def fit(self, x, y):
        self.x_hat = mean(x)
        self.y_hat = mean(y)
        self.r = self._r(x, y)
        self.b = self.r * (stdev(y)/ stdev(x))
        self.a = self.y_hat - self.b * self.x_hat

    def __predict(self, x: float) -> float:
        return self.a + self.b * x
    
    def predict(self, x: list):
        return [self.__predict(x_i) for x_i in x]
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return r2_score(y, y_pred)

    def _form(self, x):
        return self.a + self.b * x
    
    def _r(self, x: list, y:  list):
        '''
        pearson correlation coefficient
        r = Σ(x_i - xhat)(y_i - yhat) / sqrt(Σ(x_i - xhat)^2 Σ(y_i - yhat)^2)'''

        numerator, denominator_t1, denominator_t2 = 0, 0, 0
        for x_i, y_i in zip(x, y):
            numerator += (x_i - self.x_hat) * (y_i - self.y_hat)
            denominator_t1 += math.pow((x_i - self.x_hat), 2)
            denominator_t2 += math.pow((y_i - self.y_hat), 2)
        r = numerator/ math.sqrt(denominator_t1 * denominator_t2)
        return r
