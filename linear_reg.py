import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp


csv_path = 'cancer_reg.csv'
write_path = 'cancer_reg_params.csv'

data = pd.read_csv(csv_path)
cols = list(data.columns)

target = np.array(data['target_deathrate'])

non_nan_cols = data.columns[data.notna().all()].tolist()
items_to_remove = ['target_deathrate', 'binnedinc', 'geography']

non_nan_cols = list(set(non_nan_cols) - set(items_to_remove))

print(non_nan_cols)
d = {'predictor': [], 'theta': [], 'b': [], 'score': []}
thetas = []
Xs = np.array([data[col] for col in non_nan_cols]).T
# limit the number of data (n nearly equals to d)
Xs = Xs[:20]
target = target[:20]
X1s = []

for i, col in enumerate(non_nan_cols):
    reg = LinearRegression(fit_intercept=False)
    X = Xs[:, i]
    X1 = np.vstack((X, np.ones(len(X)))).T
    X1s.append(X1.T)
    reg.fit(X1, target)
    preds = reg.predict(X1)
    print(np.abs(target-preds).mean())
    thetas.append(reg.coef_)
    d['predictor'].append(col)
    d['theta'].append(reg.coef_[0])
    d['b'].append(reg.coef_[1])
    d['score'].append(reg.score(X1, target))
    # print(col, reg.coef_, reg.intercept_, reg.score(X, target))
    print(col, reg.coef_, reg.score(X1, target))

df = pd.DataFrame(data=d)
df.to_csv(write_path, index=False)

# attacK
X1s = np.vstack(X1s).T
s = cp.Variable(X1s.shape[0])
Z = np.hstack([X1s[:, 2*i:2*(i+1)].T@X1s[:, 2*i:2*(i+1)]@theta
               for i, theta in enumerate(thetas)]).T
norm = 2
objective = cp.Minimize(cp.norm(X1s.T@s - Z, norm))
constraints = []
prob = cp.Problem(objective, constraints)

result = prob.solve(solver=cp.SCS)
print(np.abs(target-s.value).mean())
print(np.min(np.abs(target-s.value)), np.max(np.abs(target-s.value)))
