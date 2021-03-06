import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

csv_path = 'cancer_reg.csv'
write_path = 'cancer_reg_params.csv'

data = pd.read_csv(csv_path)
cols = list(data.columns)

target = np.array(data['target_deathrate']).reshape(-1,1)

non_nan_cols = data.columns[data.notna().all()].tolist()
items_to_remove = ['target_deathrate', 'binnedinc','geography']

non_nan_cols = list(set(non_nan_cols) - set(items_to_remove))

print(non_nan_cols)
d = {'predictor': [], 'theta': [], 'b':[], 'score':[]}

for col in non_nan_cols:
    X = np.array(data[col]).reshape(-1, 1)
    reg = LinearRegression().fit(X, target)
    preds = reg.predict(X)
    d['predictor'].append(col)
    d['theta'].append(reg.coef_[0][0])
    d['b'].append(reg.intercept_[0])
    d['score'].append(reg.score(X, target))
    print(col, reg.coef_, reg.intercept_, reg.score(X, target))

df = pd.DataFrame(data=d)
df.to_csv(write_path, index=False)