import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp


csv_path = 'cancer_reg.csv'
write_path = 'cancer_reg_params.csv'

data = pd.read_csv(csv_path)
cols = list(data.columns)

target = np.array(data['target_deathrate']).reshape(-1, 1)

non_nan_cols = data.columns[data.notna().all()].tolist()
items_to_remove = ['target_deathrate', 'binnedinc', 'geography']

non_nan_cols = list(set(non_nan_cols) - set(items_to_remove))

print(non_nan_cols)
# d = {'predictor': [], 'theta': [], 'b': [], 'score': []}
d = {'predictor': [], 'theta': [], 'score': []}
thetas = []
Xs = np.array([data[col] for col in non_nan_cols]).T

for i, col in enumerate(non_nan_cols):
    reg = LinearRegression(fit_intercept=False)
    X = Xs[:, i].reshape(-1, 1)
    reg.fit(X, target)
    preds = reg.predict(X)
    print(np.abs(target-preds).mean())
    thetas.append(reg.coef_[0][0])
    d['predictor'].append(col)
    d['theta'].append(reg.coef_[0][0])
    # d['b'].append(reg.intercept_[0])
    d['score'].append(reg.score(X, target))
    # print(col, reg.coef_, reg.intercept_, reg.score(X, target))
    print(col, reg.coef_, reg.score(X, target))

df = pd.DataFrame(data=d)
df.to_csv(write_path, index=False)

s = cp.Variable(X.shape[0])
Z = np.array([theta*Xs[:, i].T@Xs[:, i] for i, theta in enumerate(thetas)]).T
objective = cp.Minimize(cp.norm(Xs.T@s - Z))
constraints = []
prob = cp.Problem(objective, constraints)

result = prob.solve()
print(s.value)
print(np.abs(target-s.value).mean())
