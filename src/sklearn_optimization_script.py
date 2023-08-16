# 
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Reimplementation of code found in ...
'''

import SklearnOptimisation
import pickle
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, r2_score

# search spaces

linear_space = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Lasso may not be working
lasso_space = {
    'alpha': Real(0, 3),
    'fit_intercept': Categorical([True, False]),
    'positive': Categorical([True, False]),
    'selection': Categorical([True, False])
}
rf_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(1, 10),
    'max_features': Categorical(["sqrt", "log2", None]),
    'min_samples_split': Real(1e-4, 1 - 1e-4),
    'min_samples_leaf': Real(1e-4, 1 - 1e-4),
    'n_jobs': Categorical([-1]),
}

gbt_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(1, 10),
    'max_features': Categorical(["sqrt", "log2", None]),
    'min_samples_split': Real(1e-4, 1 - 1e-4),
    'min_samples_leaf': Real(1e-4, 1 - 1e-4),
    'learning_rate': Real(1e-4, 5e-1),
    'subsample': Real(1e-4, 1 - 1e-4),
}

gp_space = {
    'alpha': Real(1e-4, 5e-1)
}

svm_space = {
    'epsilon': Real(1e-4, 5e-1),
    'C': Real(1e-4, 100)
}

# model mappings

model_dict = {
    "linear": (LinearRegression, linear_space, "Grid"),
    "lasso": (Lasso, lasso_space, "Bayesian"),
    "rf": (RandomForestRegressor, rf_space, "Bayesian"),
    "gbt": (GradientBoostingRegressor, gbt_space, "Bayesian"),
    "gp": (GaussianProcessRegressor, gp_space, "Bayesian"),
    "svm": (SVR, svm_space, "Bayesian"),
}

# data processing
class MinMaxScaler:

    def __init__(self, y):
        self.min_y = np.min(y)
        self.max_y = np.max(y)
    
    def transform(self, y):
        y_scaled = (y - self.min_y)/(self.max_y - self.min_y)
        return y_scaled

    def invert(self, y):
        y_orig = y*(self.max_y - self.min_y) + self.min_y
        return y_orig

# argument parser
def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--X', type=Path, required=True)
    parser.add_argument('--y', type=Path, required=True)    
    parser.add_argument(
        '--model_name',  
        type=str, 
        required=True, 
        description="Models allowed: linear, lasso, rf, gbt, gp, svm"
    )
    parser.add_argument('--enc_name', type=str, required=True)
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument(
        '--test_type', 
        type=str, 
        required=True, 
        description="Evaluation strategy: ncv, holdout"
    )
    parser.add_argument('--seed', type=int, required=True)
    return parser

# run optmisation
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    X_trn = np.load(args.X)
    y_trn_unscaled = np.load(args.y)
    scaler = MinMaxScaler(y_trn_unscaled)
    y_trn = scaler.transform(y_trn_unscaled)

    kf_inner = KFold(n_splits=3, random_state=8, shuffle=True)

    model, search_space, search_method = model_dict[args.model_name]

    if args.test_type == "ncv":
        print("Using nested cross validation (ncv) for model evaluation.")
        # ncv for test scores
        kf_outer = KFold(n_splits=5, random_state=8, shuffle=True)
        ncv_res = SklearnOptimisation.nCV(
            kf_inner=kf_inner,
            kf_outer=kf_outer,
            X=X_trn,
            y=y_trn,
            model=model(),
            search_space=search_space,
            search_method=search_method,
        )
        ncv_res.to_csv(f"{args.save_dir}/nCV_results.csv", index=False)

    elif args.test_type == "holdout":
        print("Using holdout for model evaluation. Default: 20%% holdout set.")
        # holdout for test scores
        cv_res, preds = SklearnOptimisation.holdout_CV(
            kf_inner=kf_inner,
            X=X_trn,
            y=y_trn,
            model=model(),
            search_space=search_space,
            search_method=search_method,
            return_predictions=True,
            seed=args.seed
        )
        json.dump(cv_res, open(f"{args.save_dir}/CV_results.json", 'w'))
        tst_pred, y_tst = preds
        np.save(f"{args.save_dir}/test_predictions.npy", tst_pred)
        np.save(f"{args.save_dir}/test_actual.npy", y_tst)

    # cv for optimisation
    final_model, final_res = SklearnOptimisation.final_opt(
        kf=kf_inner, 
        X=X_trn, 
        y=y_trn,
        model=model(),
        search_space=search_space,
        search_method=search_method,
    )
    json.dump(final_res, open(f"{args.save_dir}/final_CV_results.json", 'w'))
    pickle.dump(final_model, open(f"{args.save_dir}/final_model.sav", 'wb'))