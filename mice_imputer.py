from sklearn.base import BaseEstimator, TransformerMixin
import miceforest as mf

class mice_imputer(BaseEstimator, TransformerMixin):
    """
    Wrapper class for miceimputer around sklearn transformers to avoid error in miceimputer which requires the transform method to be called on the same dataset as the fit method was. This is a problem when trying to fit on a training set and 
    transform on a validation set within an sklearn pipeline that is called within gridsearchCV. 

    Pass any arguments as kwargs to this class from miceimputer's ImputationKernel() class, as well as from the ImputationKernel.mice() method. You can also pass arguments onto the underlying LightGBM implementation as keywords to mice. 
    Appropriate fit and transform methods will then be created such that the miceimputer.trasform method will work on new data. 
    """

    def __init__(
            self, 
            variable_parameters = None, 
            mean_match_scheme = None,
            imputation_order = "ascending",
            mice_iterations = 10, 
            lgb_objective = "regression",
            lgb_iterations = 100,
            lgb_learning_rate = .1,
            lgb_max_depth = 3,
            lgb_cat_smooth = 10,
            lgb_num_leaves = 31,
            lgb_feature_fraction_bynode = 1.0,
            lgb_min_data_in_leaf = 20
            ):
        
        self.variable_parameters = variable_parameters 
        self.mean_match_scheme = mean_match_scheme
        self.mice_iterations = mice_iterations 
        self.lgb_iterations = lgb_iterations 
        self.lgb_learning_rate = lgb_learning_rate 
        self.lgb_max_depth = lgb_max_depth 
        self.lgb_cat_smooth = lgb_cat_smooth
        self.lgb_feature_fraction_bynode = lgb_feature_fraction_bynode
        self.imputation_order = imputation_order
        self.lgb_objective = lgb_objective
        self.lgb_num_leaves = lgb_num_leaves
        self.lgb_min_data_in_leaf = lgb_min_data_in_leaf

    def fit(self, X, y=None):
        """ 
        Will first instantiate miceforest.ImputationKernel with whatever keyword args that are passed to this mice_imputer class at instantiation. Afterwards, it calls ImputationKernel.mice(), again with whatever mice() kwargs were passed at 
        instantiation, which includes kwargs which are ultimately passed on to the underlying LightGBM fitter that does the imputation. 
        """
        self.kern = mf.ImputationKernel(
            X, 
            save_models=2, 
            datasets = 1, 
            mean_match_scheme = self.mean_match_scheme,
            copy_data = True,
            imputation_order = self.imputation_order
        )

        self.kern.mice(
            variable_parameters = self.variable_parameters,
            num_iterations = self.lgb_iterations, 
            learning_rate = self.lgb_learning_rate, 
            max_depth = self.lgb_max_depth,
            cat_smooth = self.lgb_cat_smooth,
            feature_fraction_bynode = self.lgb_feature_fraction_bynode, 
            objective = self.lgb_objective,
            num_leaves = self.lgb_num_leaves,
            min_data_in_leaf = self.lgb_min_data_in_leaf
        )

        return (self)

    def transform(self, X, y = None):
        return self.kern.impute_new_data(X, copy_data=True).complete_data(inplace=False)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
