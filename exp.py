import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import perturbation as ptb
import metrics
from tabulate import tabulate
#from sklearn.metrics import mean_squared_errorx

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


def publish_regressors(Xs, target, cols, scaler,noise_func=None, **noise_kwargs):
    X1s = []
    thetas = []
    mses = []
    for i, col in enumerate(cols):
        reg = LinearRegression(fit_intercept=False)
        X = Xs[:, i]
        X1 = np.vstack((X, np.ones(len(X)))).T
        X1s.append(X1.T)
        reg.fit(X1, target)
        preds = reg.predict(X1)
        mses.append(metrics.MAE(metrics.restore(target, scaler), metrics.restore(preds,scaler)))
        theta = reg.coef_
        if noise_func is not None:
            noise = noise_func(**noise_kwargs)
            theta += noise
        thetas.append(theta)
        # print(col, reg.coef_, reg.score(X1, target),
        #       np.abs(target-preds).mean())
    X1s = np.vstack(X1s).T
    return X1s, thetas,mses


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
    print(np.min(target), np.max(target), len(cols))
    # fn = ptb.normal
    # kwargs = {'sigma': 0.01, 'size': 2}
    # fn = ptb.trunc_normal
    # kwargs = {'b': 0.02, 'scale': 0.01, 'size': 2}
    fn = ptb.laplace
    kwargs = {'scale': 0.0001, 'size': 2}
    # Xs, thetas = publish_regressors(Xs, target, cols, fn, **kwargs)
    Xs, thetas, _ = publish_regressors(Xs, target, cols, scaler)

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
    mse_orig = []
    std_orig = []
    for n in ns:
        maes = []
        mses = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            Xs, thetas, mse = publish_regressors(Xs, target, cols, scaler)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
            if n==20:
                mses.append(mse)
        means.append(round(np.mean(maes),2))
        stds.append(round(np.std(maes),3))
        if n==20:
            mse_orig.append(np.mean(mses, axis=0))
            std_orig.append(np.std(mses, axis=0))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(ns, means, stds, solid_capstyle='projecting', capsize=5,linestyle='None', marker='^')
    plt.xlabel('Dataset Size')
    plt.ylabel('MAE')
    plt.xscale('log')
    plt.savefig('result/non_priv_n.png')
    plt.show()
    plt.close()
    print(tabulate({"Dataset Size": ns, "MAE": means, "Std Deviation": stds}, headers="keys", tablefmt="latex"))


    n = 20
    # laplace noise
    scales = [1, 0.7,0.3, 0.1, 0.07,0.03,0.01,0.007,0.003, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
    print(scales)
    reps = 10
    means_lap = []
    stds_lap = []
    mses_ln_mean = []
    mses_ln_std = []
    fn = ptb.laplace
    for scale in scales:
        maes = []
        mses = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'scale': scale, 'size': 2}
            Xs, thetas, mse = publish_regressors(Xs, target, cols, scaler, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
            mses.append(mse)
        means_lap.append(round(np.mean(maes),2))
        stds_lap.append(round(np.std(maes),3))
        mses_ln_mean.append(np.mean(mses, axis=0))
        mses_ln_std.append(np.std(mses, axis=0))
        print(np.mean(maes), np.std(maes), len(mses_ln_mean[0]))
    plt.errorbar(scales, means_lap, stds_lap, solid_capstyle='projecting', capsize=3,linestyle='None', marker='^')
    plt.xlabel('Scale')
    plt.ylabel('MAE')
    plt.xscale('log')
    plt.savefig('result/lap.png')
    plt.show()
    plt.close()
    print('LAPLACE')
    print(tabulate({"Scale": scales, "MAE": means_lap, "Std Deviation": stds_lap}, headers="keys", tablefmt="latex"))

    # gaussian noise
    sigmas = [1, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
    reps = 10
    means_gauss = []
    stds_gauss = []
    mses_gn_mean = []
    mses_gn_std = []
    fn = ptb.normal
    for sigma in sigmas:
        maes = []
        mses = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'sigma': sigma, 'size': 2}
            Xs, thetas, mse = publish_regressors(Xs, target, cols, scaler, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
            mses.append(mse)
        means_gauss.append(round(np.mean(maes),2))
        stds_gauss.append(round(np.std(maes),3))
        mses_gn_mean.append(np.mean(mses, axis=0))
        mses_gn_std.append(np.std(mses, axis=0))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(scales, means_gauss, stds_gauss,solid_capstyle='projecting', capsize=3, linestyle='None', marker='^')
    plt.xlabel('Sigma')
    plt.ylabel('MAE')
    plt.xscale('log')
    plt.savefig('result/gauss.png')
    plt.show()
    plt.close()
    print('Gaussian')
    print(tabulate({"Sigma": sigmas, "MAE": means_gauss, "Std Deviation": stds_gauss}, headers="keys", tablefmt="latex"))

    # truc normal noise
    scales = [1, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
    reps = 10
    means_trunc = []
    stds_trunc = []
    mses_tn_mean = []
    mses_tn_std = []
    fn = ptb.trunc_normal
    for scale in scales:
        maes = []
        mses = []
        for _ in range(reps):
            Xs, target, cols, scaler = read_data(True, n, normalize=True)
            kwargs = {'b': 1/n, 'scale': scale, 'size': 2}
            Xs, thetas, mse = publish_regressors(Xs, target, cols, scaler, fn, **kwargs)
            rec = attack(Xs, thetas)
            maes.append(metrics.MAE(target, rec, scaler))
            mses.append(mse)
        means_trunc.append(round(np.mean(maes),2))
        stds_trunc.append(round(np.std(maes),3))
        mses_tn_mean.append(np.mean(mses, axis=0))
        mses_tn_std.append(np.std(mses, axis=0))
        print(np.mean(maes), np.std(maes))
    plt.errorbar(scales, means_trunc, stds_trunc, solid_capstyle='projecting', capsize=3,linestyle='None', marker='^')
    plt.xlabel('Scale')
    plt.ylabel('MAE')
    plt.xscale('log')
    plt.savefig('result/truc_normal.png')
    plt.show()
    plt.close()
    print('Truncated Normal')
    print(tabulate({"Scale": scales, "MAE": means_trunc, "Std Deviation": stds_trunc}, headers="keys", tablefmt="latex"))


    def plot(final_mean, final_std,s,scales, mse_orig, std_orig):
        import matplotlib.colors as colors

        colors_list = list(colors._colors_full_map.values())

        plt.figure()
        #plt.plot(list(range(27)), mse_orig[0],color=colors_list[0], label='no noise')

        for i in [0,3,6]:

            m = final_mean[i]/mse_orig[0] # we are actually calculating MAE
            plt.plot(list(range(27)), m, color=colors_list[i + 150], label=str(scales[i]))

        plt.xlabel('Regressors based on feature on the x axis')
        plt.ylabel('MAE')
        plt.legend(loc="upper left")
        plt.savefig('result/regressors_{}.png'.format(s))
        plt.show()
        plt.close()

    plot(mses_ln_mean, mses_ln_std,'laplace',scales, mse_orig, std_orig)
    plot(mses_gn_mean, mses_gn_std, 'gaussian', sigmas, mse_orig, std_orig)
    plot(mses_tn_mean, mses_tn_std, 'truncated_normal', scales, mse_orig, std_orig)
