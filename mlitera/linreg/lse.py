import numpy as np

class LSE(object):
    
    def fit(self, X, y):
        N = len(X)
        theta1 = ( sum(X * y) - (N * X.mean() * y.mean()) ) / sum((X - X.mean()) ** 2)
        theta0 = y.mean() - theta1 * X.mean()
        self.thetas = np.array([theta0, theta1])
        return self.thetas
    
    def score(self, ytrue, yreg):
        SST = sum((ytrue - ytrue.mean()) ** 2)
        SSR = sum((ytrue - yreg) ** 2)
        return 1 - SSR/ SST
    
    def predict(self, X):
        y = self.thetas[0] + self.thetas[1] * X
        return y
        

if __name__ == '__main__':
    X = np.array([27, 46, 73, 40, 30, 28, 46, 59])
    y = np.array([ 5, 10, 20,  8,  4,  6, 12, 15])
    m = LSE()
    coef = m.fit(X, y)
    print(coef)
    print(m.predict(X))
    print(m.score(y, m.predict(X)))