import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp
import perturbation as ptb
import argparse

csv_path = 'cancer_reg.csv'
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
        #print("intercepts: ",reg.intercept_)
        if noise_func is not None:
            noise = noise_func(**noise_kwargs)
            theta += noise
        thetas.append(theta)
        #print(col, reg.coef_, reg.score(X1, target),
        #      np.abs(target-preds).mean())
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
    parser = argparse.ArgumentParser(description='Input values')
    parser.add_argument('--cutoff','-c',type=bool)
    parser.add_argument('--cutoff_n','-cn',type=int)
    parser.add_argument('--ntype','-nt',help='Noise type',type=int)
    parser.add_argument('--wpath', '-wp', help='Write path',type=str)
    parser.add_argument('--noise_params', '-np',help='Noise parameters',nargs='+')
    args = parser.parse_args()

    means = []
    mins = []
    maxs = []
    cutoff = args.cutoff
    cutoff_n = args.cutoff_n

    noises = {0:ptb.normal, 1:ptb.laplace, 2:ptb.trunc_normal,-1:None}
    ntype = args.ntype
    fn = noises[ntype]
    kwargs = {}
    if fn is not None:
        kwargs_dict = {0:(['sigma', 'size', 'mu'],[float,  int,float]),
                       1:(['scale', 'size', 'loc'],[float,int,float]),
                       2: (['b', 'scale', 'size', 'loc', 'a'],[float,float,int,float,float])}
        kwargs_list = kwargs_dict[ntype][0]
        kwargs_type = kwargs_dict[ntype][1]
        for j,l_i in enumerate(args.noise_params):
            kwargs[kwargs_list[j]] = kwargs_type[j](l_i)
        print('kwargs : ', kwargs)

    for i in range(10):
        Xs, target, cols = read_data(cutoff, cutoff_n)
        Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
        rec = attack(Xs, thetas)
        means.append(np.abs(target-rec).mean())
        mins.append(np.min(np.abs(target-rec)))
        maxs.append(np.max(np.abs(target-rec)))

    mean = np.mean(means)
    std = np.std(means)
    print(means, mins, maxs)
    with open(args.wpath, 'w+') as f:
        f.write(str(args) + '\n')
        f.write(str(mean) + ' ' + str(std) + '\n')
