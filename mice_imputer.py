import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.impute import MissingIndicator, SimpleImputer
import matplotlib.pyplot as plt
import miceforest as mf
from miceforest import mean_match_default
import seaborn as sns
from lightgbm import LGBMClassifier
import inspect 
import warnings
import scipy.stats as stats
from tempfile import mkdtemp
from shutil import rmtree

class mice_imputer(BaseEstimator, TransformerMixin):
    """
    Wrapper class for miceimputer around sklearn transformers to avoid error in miceimputer which requires the transform method to be called on the same dataset as the fit method was. This is a problem when trying to fit on a training set and 
    transform on a validation set within an sklearn pipeline that is called within gridsearchCV. 

    Pass any arguments as kwargs to this class from miceimputer's ImputationKernel() class, as well as from the ImputationKernel.mice() method. You can also pass arguments onto the underlying LightGBM implementation as keywords to mice. 
    Appropriate fit and transform methods will then be created such that the miceimputer.trasform method will work on new data. 
    """

    def __init__(self, variable_parameters=None, **kwargs):
        self.all_kwargs = kwargs
        self.lgb_args = {"num_iterations", "learning_rate", "num_leaves",
                         "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf",
                         "bagging_fraction", "colsample_bytree", "colsample_bynode",
                         "lambda_l1", "lambda_l2", "min_split_gain", "cat_smooth"}
        self.lgb_args = self.__arg_intersect(
            self.all_kwargs, self.lgb_args, right_fn=False)
        self.inst_args = self.__arg_intersect(
            self.all_kwargs, mf.ImputationKernel)
        self.mice_args = self.__arg_intersect(
            self.all_kwargs, mf.ImputationKernel.mice)
        self.variable_parameters = variable_parameters
        self.kern = []

        self.invalid = set(self.all_kwargs.keys()).difference(set(
            self.inst_args.keys()).union(set(self.lgb_args.keys()), set(self.mice_args.keys())))

        if len(self.invalid) > 0:
            warnings.warn(
                "Invalid **kwargs will be ignored:{}".format(self.invalid))

    def __arg_intersect(self, kwargs_dict, right, right_fn=True):
        right = inspect.getfullargspec(right).args if right_fn else right
        inter = kwargs_dict.keys() & right
        out_dict = {key: kwargs_dict[key] for key in inter}

        return out_dict

    def __warn_clean(message, category, filename, lineno, file=None, line=None):
        return ("%s:%s %s: %s\n") % (filename, lineno, category.__name__, message)
    warnings.formatwarning = __warn_clean

    def __merge_dict(self, *args):
        base = dict()
        for i in args:
            base.update(i)
        return (base)

    # Mandatory for sklearn api
    def get_params(self, deep=True):
        return self.__merge_dict(self.lgb_args, self.inst_args, self.mice_args)

    # Mandatory for sklearn api
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None):
        """ 
        Will first instantiate miceforest.ImputationKernel with whatever keyword args that are passed to this mice_imputer class at instantiation. Afterwards, it calls ImputationKernel.mice(), again with whatever mice() kwargs were passed at 
        instantiation, which includes kwargs which are ultimately passed on to the underlying LightGBM fitter that does the imputation. 
        """
        self.kern = mf.ImputationKernel(X, save_models=2, **self.inst_args)
        self.kern.mice(variable_parameters=self.variable_parameters,
                       **self.mice_args, **self.lgb_args)
        return (self)

    def transform(self, X, y=None):
        return self.kern.impute_new_data(X, copy_data=True).complete_data(inplace=False)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
