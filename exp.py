import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp
import perturbation as ptb


csv_path = 'cancer_reg.csv'
write_path = 'cancer_reg_params.csv'
rng = np.random.default_rng(12345)


def read_data(cutoff=False, n=0):
    data = pd.read_csv(csv_path)
    print(data['target_deathrate'].describe())
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
    return Xs, target, non_nan_cols


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
        print(col, reg.coef_, reg.score(X1, target),
              np.abs(target-preds).mean())
    X1s = np.vstack(X1s).T
    return X1s, thetas


def attack(Xs, thetas, norm=2):
    s = cp.Variable(Xs.shape[0])
    Z = np.hstack([Xs[:, 2*i:2*(i+1)].T@Xs[:, 2*i:2*(i+1)]@theta
                   for i, theta in enumerate(thetas)]).T
    objective = cp.Minimize(cp.norm(Xs.T@s - Z, norm))
    constraints = [s >= 0, s <= 400]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    assert s.value is not None
    return s.value


if __name__ == '__main__':
    cutoff = False
    cutoff_n = 20
    Xs, target, cols = read_data(cutoff, cutoff_n)
    # fn = ptb.normal
    # kwargs = {'sigma': 0.01, 'size': 2}
    # fn = ptb.trunc_normal
    # kwargs = {'b': 0.02, 'scale': 0.01, 'size': 2}
    fn = ptb.laplace
    kwargs = {'scale': 0.0001, 'size': 2}
    Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
    rec = attack(Xs, thetas)
    print(np.abs(target-rec).mean())
    print(np.min(np.abs(target-rec)), np.max(np.abs(target-rec)))
