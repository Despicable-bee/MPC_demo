# Standard libs
import sys

# Third party libs
import jax.numpy as np
import jax
import numpy as onp
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)




# Local libs

# * GLOBAL SETUP ---------------------------------------------------------------

logger = logging.getLogger('JAX SGD')

class Bootstrap:
    def __init__(self, seed=123):
        """
        bootstrap 1d array
        usage:
        xs = np.arrange(100)
        bs = Bootstrap(seed=123)
        for idx in bs.bootstrap(xs, group_size=50, n_boots=10):
            print(xs[idx].mean())
        """
        self.rng = onp.random.RandomState(seed)

    def bootstrap(self, xs, group_size=100, n_boots=100):
        """
        input:
            xs 1d np.array
            group_size: number of values in each bootstrap iteration
            n_boots: how many bootstrap groups
        output:
            iterator: bootstrapped
        """
        xs = onp.array(xs)
        total_size = xs.shape[0]
        logger.info("Total size for bootstrap: {}".format(total_size))

        if group_size > total_size:
            raise ValueError("Group size > input array size")
        
        for _ in range(n_boots):
            idx = self.rng.randint(0, total_size, group_size)
            yield idx
        

# * LOSS FUNCTION --------------------------------------------------------------

@jax.jit
def loss_function(params, x, y):
    """
    Root mean square loss function:

    input:
        - params: a list [w, b] where w are the weights and b is the bias term
        - x: input data for training (np.array)
        - y: target data (np.array)
    
    return:
        - RMSE value (float)
    """
    # with jax.disable_jit():
    #     print("Loss function inputs:")
    #     print("Params:")
    #     print("{}, {}".format(params[0], params[1]))
    #     print("x:")
    #     print(x)
    #     print("y:")
    #     print(y)

    predict = x.dot(params[0]) + params[1]
    deviation = y - predict
    squared_deviation = deviation ** 2
    mean_squared_deviation = squared_deviation.mean()
    loss = np.sqrt(mean_squared_deviation)
    #print(loss)
    return loss

# * SGD ------------------------------------------------------------------------

class SGD():
    """
    This is a lineare model solver using minibatch stochastic gradient descent

    usage:
        # some test data
        X = 10 * onp.random.random((1000, 2))
        y = X.dot([3,4]) + onp.random.random(1000) + 5

        # model fitting
        lm = SGD(n_epoch=10000, learning_rate=0.001)
        lm.fit(X,y)
    """
    def __init__(self, learning_rate=1e-3, n_epoch=1000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.losses = onp.zeros(n_epoch)
        self.coef_ = None
        self.intercept_ = None
        self.gradients = None
        self._iter = 0

    def fit(self, X, y):
        if X.ndim != 2:
            raise ValueError("X must have 2 dimensions")
        
        self.__InitParams__(X)
        bootstrap = Bootstrap()
        subsets = bootstrap.bootstrap(X, group_size=100, n_boots=self.n_epoch)
        loss_gradient = jax.grad(loss_function)

        for i in range(self.n_epoch):
            self._iter += 1
            train_idx = next(subsets)
            X_train, y_train = X[train_idx], y[train_idx]

            loss = loss_function([self.coef_, self.intercept_], X_train, 
                    y_train)
            
            self.losses[i] = loss
            self.gradients = loss_gradient([self.coef_, self.intercept_], 
                    X_train, y_train)

            print(self.gradients)
            sys.exit()

            self.__update__()
            if self._iter % (self.n_epoch//10):
                logger.info("{} epoch - Loss: {}".format(self._iter, loss))
            
    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
    
    def __InitParams__(self, X):
        # Initialise weights and bias terms
        self.coef_ = onp.random.randn(X.shape[1])
        self.intercept_ = onp.random.randn(1)
        self._iter = 0
    
    def __update__(self):
        # Update weight and bias terms with gradients
        # gradient[0]: gradients for the coefficients
        # gradient[1]: gradients for the bias terms

        self.coef_ -= self.gradients[0] * self.learning_rate
        self.intercept_ -= self.gradients[1] * self.learning_rate


if __name__ == '__main__':
    X = 10 * onp.random.random((1000,2))
    y = X.dot([3,4]) + onp.random.random(1000) + 5

    lm = SGD(n_epoch=5000, learning_rate=0.01)
    lm.fit(X,y)
    
    plt.plot(lm.losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()
