import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import perturbation as ptb
import metrics


csv_path = 'cancer_reg.csv'
write_path = 'cancer_reg_params.csv'
rng = np.random.default_rng(12345)


def read_data(cutoff=False, n=0, normalize=False):
    data = pd.read_csv(csv_path)
    target = np.array(data['target_deathrate'])

    non_nan_cols = data.columns[data.notna().all()].tolist()
    items_to_remove = ['target_deathrate', 'binnedinc', 'geography']

    non_nan_cols = list(set(non_nan_cols) - set(items_to_remove))

    Xs = np.array([data[col] for col in non_nan_cols]).T

    if cutoff:
        assert n <= len(target)
        indices = rng.choice(len(target), n)
        Xs = Xs[indices]
        target = target[indices]

    target_scaler = None
    if normalize:
        scaler = preprocessing.StandardScaler().fit(Xs)
        Xs = scaler.transform(Xs)
        target_scaler = preprocessing.StandardScaler()
        target_scaler.fit(target.reshape(-1, 1))
        target = target_scaler.transform(target.reshape(-1, 1))[:, 0]

    return Xs, target, non_nan_cols, target_scaler


def publish_regressors(Xs, target, cols, noise_func=None, **noise_kwargs):
    X1s = []
    thetas = []
    for i, col in enumerate(cols):
        reg = LinearRegression(fit_intercept=False)
        X = Xs[:, i]
        X1 = np.vstack((X, np.ones(len(X)))).T
        X1s.append(X1.T)
        reg.fit(X1, target)
        preds = reg.predict(X1)
        theta = reg.coef_
        if noise_func is not None:
            noise = noise_func(**noise_kwargs)
            theta += noise
        thetas.append(theta)
        # print(col, reg.coef_, reg.score(X1, target),
        #       np.abs(target-preds).mean())
    X1s = np.vstack(X1s).T
    return X1s, thetas


def attack(Xs, thetas, norm=2):
    s = cp.Variable(Xs.shape[0])
    Z = np.hstack([Xs[:, 2*i:2*(i+1)].T@Xs[:, 2*i:2*(i+1)]@theta
                   for i, theta in enumerate(thetas)]).T
    objective = cp.Minimize(cp.norm(Xs.T@s - Z, norm))
    constraints = []
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    assert s.value is not None
    return s.value


def attack_dual(Xs, thetas, dnorm=2):
    Z = np.hstack([Xs[:, 2*i:2*(i+1)].T@Xs[:, 2*i:2*(i+1)]@theta
                   for i, theta in enumerate(thetas)]).T
    nu = cp.Variable(Z.shape)
    objective = cp.Maximize(Z.T@nu)
    constraints = [cp.norm(nu, 2) <= 1, Xs @ nu == 0]
    prob = cp.Problem(objective, constraints)
    res = prob.solve(solver=cp.SCS)
    s = np.linalg.lstsq(Xs.T, Z+res*nu.value, rcond=None)
    return s[0]


if __name__ == '__main__':
    cutoff = True
    cutoff_n = 20
    Xs, target, cols, scaler = read_data(cutoff, cutoff_n, normalize=True)
    print(np.min(target), np.max(target))
    # fn = ptb.normal
    # kwargs = {'sigma': 0.01, 'size': 2}
    # fn = ptb.trunc_normal
    # kwargs = {'b': 0.02, 'scale': 0.01, 'size': 2}
    fn = ptb.laplace
    kwargs = {'scale': 0.0001, 'size': 2}
    # Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
    Xs, thetas = publish_regressors(Xs, target, cols)
    rec = attack(Xs, thetas)
    print('MAE:', metrics.MAE(target, rec, scaler))
    print('min:', metrics.absmin(target, rec, scaler))
    print('max:', metrics.absmax(target, rec, scaler))
    rec = attack_dual(Xs, thetas)
    print('MAE:', metrics.MAE(target, rec, scaler))
    print('min:', metrics.absmin(target, rec, scaler))
    print('max:', metrics.absmax(target, rec, scaler))

    # experiments
    # number of n for non-private
    ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 3047] 
    reps = 50
    means = []
    stds = []
    for n in ns:
        maes = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            Xs, thetas = publish_regressors(Xs, target, cols)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
        means.append(np.mean(maes))
        stds.append(np.std(maes))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(ns, means, stds, linestyle='None', marker='^')
    plt.xscale('log')
    plt.savefig('result/non_priv_n.png')
    plt.close()

    n = 20
    # laplace noise
    scales = [10**(-i) for i in range(10)]
    reps = 5
    means = []
    stds = []
    fn = ptb.laplace
    for scale in scales:
        maes = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'scale': scale, 'size': 2}
            Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
        means.append(np.mean(maes))
        stds.append(np.std(maes))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(scales, means, stds, linestyle='None', marker='^')
    plt.xscale('log')
    plt.savefig('result/lap.png')
    plt.close()

    # gaussian noise
    sigmas = [10**(-i) for i in range(10)]
    reps = 5
    means = []
    stds = []
    fn = ptb.normal
    for sigma in sigmas:
        maes = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'sigma': sigma, 'size': 2}
            Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
        means.append(np.mean(maes))
        stds.append(np.std(maes))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(scales, means, stds, linestyle='None', marker='^')
    plt.xscale('log')
    plt.savefig('result/gauss.png')
    plt.close()

    # truc normal noise
    scales = [10**(-i) for i in range(10)]
    reps = 5
    means = []
    stds = []
    fn = ptb.trunc_normal
    for scale in scales:
        maes = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'b': 1/n, 'scale': scale, 'size': 2}
            Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
        means.append(np.mean(maes))
        stds.append(np.std(maes))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(scales, means, stds, linestyle='None', marker='^')
    plt.xscale('log')
    plt.savefig('result/truc_normal.png')
    plt.close()
