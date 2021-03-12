import numpy as np
from scipy.stats import truncnorm


np.random.seed(seed=12345)
rng = np.random.default_rng(12345)


def normal(sigma, size, mu=0):
    return rng.normal(mu, sigma, size)


def laplace(scale, size, loc=0):
    return rng.laplace(loc, scale, size)


def trunc_normal(b, scale, size, loc=0, a=0):
    a = -b if a == 0 else a
    return truncnorm.rvs(a, b, loc, scale, size)


if __name__ == '__main__':
    tmp = normal(1, 2)
    print(tmp, tmp.shape)
    tmp = laplace(1, 2)
    print(tmp, tmp.shape)
    tmp = laplace(1, 2)
    print(tmp, tmp.shape)
    tmp = trunc_normal(0.1, 1, 2)
    print(tmp, tmp.shape)
    tmp = trunc_normal(0.1, 1, 2)
    print(tmp, tmp.shape)
