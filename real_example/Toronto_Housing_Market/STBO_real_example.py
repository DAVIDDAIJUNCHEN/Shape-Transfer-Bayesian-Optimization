import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# STBO real example: XGBOOST on predicting Tornodo house price 
scaler = StandardScaler()

x= (3, 0.05, 2, 0.5, 0.3)
max_depth, lr, max_delta_step, colsample_bytree, subsample = x

params =  {
            'objective': 'reg:squarederror',
            'max_depth': int(max_depth),
            'learning_rate': lr,
            'max_delta_step': int(max_delta_step),
            'colsample_bytree': colsample_bytree,
            'subsample' : subsample
            }

X_train = pd.read_csv("./Tornodo_X.csv")
y_train = pd.read_csv("./Tornodo_Y.csv")

X_train = X_train.values
y_train = y_train.values

y_train = y_train[:,3]
y_train = y_train.reshape(-1, 1)
y_train = scaler.fit_transform(y_train)

data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

cv_results = xgb.cv(params=params, 
                    dtrain=data_dmatrix, 
                    nfold=3, 
                    seed=3,
                    num_boost_round=50,
                    early_stopping_rounds=50,
                    metrics='rmse')

results = 10 - cv_results['test-rmse-mean'].min()

print("XGB results: ", results)
