from darts.models import NBEATSModel, RandomForest, XGBModel
from darts.metrics import rmse
from darts.utils.likelihood_models import QuantileRegression
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# k fold cross validation could be used for tree based methods, expensive for NBEATS
# TODO remove n_jobs, not a parameter of the model for xgboost and random forest
# TODO add early stopping for all models, especially NBEATS

class HyperparameterOptimizationNBEATS:
    """
    Class for hyperparameter optimization of NBEATS model

    Args:
        model (NBEATSModel): NBEATS model
        input_length (int): Input length
        output_length (int): Output length
        train_target (TimeSeries): Training target
        train_past_cov (TimeSeries): Training past covariates
        val_target_input (TimeSeries): Validation target
        val_past_cov (TimeSeries): Validation past covariates
        seed (int, optional): Random seed. Defaults to 42.
    """
    def __init__(self, train_target, train_past_cov, val_target, val_input, val_past_cov, seed=42):
        self.output_length = len(train_target[0]) - len(val_input[0])
        self.train_target = train_target
        self.train_past_cov = train_past_cov
        self.val_target = val_target
        self.val_input = val_input
        self.val_past_cov = val_past_cov
        self.seed = seed
        self.space = [
            Integer(2, 5, name='num_stacks'),
            Integer(4, 10, name='num_blocks'),
            Integer(64, 256, name='layer_width'),
            Integer(50, 300, name='n_epochs'),
            Integer(1, 3, name='nr_epochs_val_period'),
            Integer(4, len(val_input[0]), name='input_length')]

    def objective_nbeats(self, params):
        """ 
        Optimization function for hyperparameter tuning

        Args:
            params (tuple): hyperparameters to optimize
            train_target (TimeSeries): training target
            train_past_cov (TimeSeries): training past covariates
            test_target (TimeSeries): validation target
            test_past_cov (TimeSeries): validation past covariates

        Returns:
            float: validation loss
        """
        # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
        # a period of 5 epochs (`patience`)
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=5,
            min_delta=0.005,
            mode='min',
        )

        n_stacks, n_blocks, layer_width, epochs, val_wait, input_length = map(int, params)
        quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        model = NBEATSModel(input_chunk_length=input_length, 
                            output_chunk_length=self.output_length,
                            num_stacks=n_stacks,
                            num_blocks=n_blocks,
                            layer_widths=layer_width,
                            n_epochs=epochs,
                            likelihood=QuantileRegression(quantiles),
                            optimizer_kwargs={"lr": 1e-3},
                            pl_trainer_kwargs={"callbacks": [my_stopper]},
                            generic_architecture=True,
                            trend_polynomial_degree=2,
                            random_state=self.seed,
                            nr_epochs_val_period=val_wait,
                            )
        model.fit(series=self.train_target,
                  past_covariates=self.train_past_cov,
                  epochs=epochs,
                  verbose=True)
        preds_val = model.predict(n=self.output_length,
                                series=self.val_input,
                                past_covariates=self.val_past_cov,)
        
        return np.mean([rmse(target[-self.output_length:], pred) for target, pred in zip(self.val_target, preds_val)])
    
    
    def optimize(self, n_calls=50):
        """
        Optimize hyperparameters using Bayesian optimization and Gaussian processes.

        Args:
            n_calls (int, optional): Number of calls to the objective function. Defaults to 50.
        
        Returns:
            skopt.OptimizeResult: Optimization result
        """
        # Wrap objective function to use named arguments
        @use_named_args(self.space)
        
        def objective(num_stacks, num_blocks, layer_width, n_epochs, nr_epochs_val_period, input_length):
            """
            Objective function for hyperparameter tuning

            Args:
                num_stacks (int): Number of stacks
                num_blocks (int): Number of blocks
                layer_width (int): Width of the layers
                n_epochs (int): Number of epochs
                nr_epochs_val_period (int): Number of epochs between validation
            
            Returns:
                float: validation loss
            """
            params = (num_stacks, num_blocks, layer_width, n_epochs, nr_epochs_val_period, input_length)
            return self.objective_nbeats(params)
        
        result = gp_minimize(objective, self.space, n_calls=n_calls)
        return result

# TODO add support for past covariates in tree based models 

class HyperparameterOptimizationRandomForest:
    """
    Class for hyperparameter optimization of a Random Forest Model
    using k fold cross validation and Bayesian optimization.

    Args:
        train_target (TimeSeries): Training target
        train_future_cov (TimeSeries): Training past covariates
        val_target (TimeSeries) : Validation target
        val_input (TimeSeries): Validation target input for prediction
        val_future_cov (TimeSeries): Validation past covariates
        seed (int, optional): Random seed. Defaults to 42.
    """
    def __init__(self, train_target, train_future_cov, val_target, val_input, val_future_cov, seed=42):

        self.output_length = len(train_target[0]) - len(val_input[0])
        self.train_target = train_target
        self.train_future_cov = train_future_cov
        self.val_target = val_target
        self.val_future_cov = val_future_cov
        self.val_input = val_input
        self.seed = seed
        self.space = [
            Integer(2, 10, name='max_depth'),
            Integer(2, 10, name='min_samples_split'),
            Integer(1, 5, name='min_samples_leaf'),
            Integer(10, 500, name='n_estimators'),
            Integer(1, 3, name='n_jobs'),
            Integer(3, 12, name='lags'),
            Integer(-len(val_input[0]), -1, name='lags_future_covariates')]
    
    def objective_rf(self, params):
        """ 
        Optimization function for hyperparameter tuning

        Args:
            params (tuple): hyperparameters to optimize
        Returns:
            float: validation loss
        """
        max_depth, min_samples_split, min_samples_leaf, n_estimators, n_jobs, lags, lags_future_covariates = map(int, params)
        
        model = RandomForest(   output_chunk_length=self.output_length,
                                lags=lags,
                                lags_future_covariates=[k for k in range(lags_future_covariates, 1)],
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                n_estimators=n_estimators,
                                n_jobs=n_jobs,
                                random_state=self.seed)
        
        model.fit(series=self.train_target, 
                  future_covariates=self.train_future_cov)
        
        preds_val = model.predict(n=self.output_length, 
                                  series=self.val_input,
                                  future_covariates=self.val_future_cov)
        
        return np.mean([rmse(target[-self.output_length:], pred) for target, pred in zip(self.val_target, preds_val)])

    def optimize(self, n_calls=50):
        """
        Optimize hyperparameters using Bayesian optimization and Gaussian processes.

        Args:
            n_calls (int, optional): Number of calls to the objective function. Defaults to 50.
        
        Returns:
            skopt.OptimizeResult: Optimization result
        """
        # Wrap objective function to use named arguments
        @use_named_args(self.space)
        def objective(max_depth, min_samples_split, min_samples_leaf, n_estimators, n_jobs, lags, lags_future_covariates):
            """
            Objective function for hyperparameter tuning

            Args:
                max_depth (int): Maximum depth of the tree
                min_samples_split (int): Minimum number of samples required to split an internal node
                min_samples_leaf (int): Minimum number of samples required to be at a leaf node
                n_estimators (int): Number of trees in the forest
                n_jobs (int): Number of jobs to run in parallel
                lags (int): Number of lags
                lags_future_covariates (int): Number of lags for future covariates
            Returns:
                float: validation loss
            """
            params = (max_depth, min_samples_split, min_samples_leaf, n_estimators, n_jobs, lags, lags_future_covariates)
            return self.objective_rf(params)
        
        result = gp_minimize(objective, self.space, n_calls=n_calls)
        return result
    
# TODO check why this doesnt converge 
    
class HyperparameterOptimizationXGB:
    """
    Class for hyperparameter optimization of a Random Forest Model
    using k fold cross validation and Bayesian optimization.

    Args:
        train_target (TimeSeries): Training target
        train_future_cov (TimeSeries): Training past covariates
        val_target (TimeSeries) : Validation target
        val_input (TimeSeries): Validation target input for prediction
        val_future_cov (TimeSeries): Validation past covariates
        seed (int, optional): Random seed. Defaults to 42.
    """
    def __init__(self, train_target, train_future_cov, val_target, val_input, val_future_cov, seed=42):

        self.output_length = len(train_target[0]) - len(val_input[0])
        self.train_target = train_target
        self.train_future_cov = train_future_cov
        self.val_target = val_target
        self.val_future_cov = val_future_cov
        self.val_input = val_input
        self.seed = seed
        self.space = [
            Integer(2, 10, name='max_depth'),
            Integer(10, 500, name='n_estimators'),
            Integer(1, 3, name='n_jobs'),
            Integer(3, 12, name='lags'),
            Integer(-len(val_input[0]), -1, name='lags_future_covariates'),
            Real(0.001, 0.5, name='learning_rate')]
    
    def objective_xgb(self, params):
        """ 
        Optimization function for hyperparameter tuning

        Args:
            params (tuple): hyperparameters to optimize

        Returns:
            float: validation loss
        """
        max_depth, n_estimators, n_jobs, lags, lags_future_covariates, learning_rate = map(int, params)
        
        model = XGBModel(output_chunk_length=self.output_length,
                        lags=lags,
                        lags_future_covariates=[k for k in range(lags_future_covariates, 1)],
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        random_state=self.seed,
                        learning_rate=learning_rate)

        model.fit(series=self.train_target, 
                  future_covariates=self.train_future_cov)
        
        preds_val = model.predict(n=self.output_length, 
                                  series=self.val_input,
                                  future_covariates=self.val_future_cov)
        
        return np.mean([rmse(target[-self.output_length:], pred) for target, pred in zip(self.val_target, preds_val)])

    def optimize(self, n_calls=50):
        """
        Optimize hyperparameters using Bayesian optimization and Gaussian processes.

        Args:
            n_calls (int, optional): Number of calls to the objective function. Defaults to 50.
        
        Returns:
            skopt.OptimizeResult: Optimization result
        """
        # Wrap objective function to use named arguments
        @use_named_args(self.space)
        def objective(max_depth, n_estimators, n_jobs, lags, lags_future_covariates, learning_rate):
            """
            Objective function for hyperparameter tuning

            Args:
                max_depth (int): Maximum depth of the tree
                min_samples_split (int): Minimum number of samples required to split an internal node
                min_samples_leaf (int): Minimum number of samples required to be at a leaf node
                n_estimators (int): Number of trees in the forest
                n_jobs (int): Number of jobs to run in parallel
                lags (int): Number of lags
                lags_future_covariates (int): Number of lookbacks included in model
            
            Returns:
                float: validation loss
            """
            params = (max_depth, n_estimators, n_jobs, lags, lags_future_covariates, learning_rate)
            return self.objective_xgb(params)
        
        result = gp_minimize(objective, self.space, n_calls=n_calls)
        return result

# global class for hyperparameter optimization using the models above, just made this because it was easier to use in the notebook
class HyperparameterOptimization:
    """
    Optimizes hyperparameters for a given model

    Args:
        model (str): Model to optimize (must be either 'nbeats', 'rf' or 'xgb')
        train_target (TimeSeries): Training target
        train_past_cov (TimeSeries): Training past covariates
        train_future_cov (TimeSeries): Training future covariates
        val_target (TimeSeries) : Validation target
        val_input (TimeSeries): Validation target input for prediction
        val_past_cov (TimeSeries): Validation past covariates
        val_future_cov (TimeSeries): Validation future covariates
        seed (int, optional): Random seed. Defaults to 42.
    """
    def __init__(self, train_target, train_past_cov, train_future_cov, val_target, val_input, val_past_cov, val_future_cov, seed=42):
        self.train_target = train_target
        self.train_past_cov = train_past_cov
        self.train_future_cov = train_future_cov
        self.val_target = val_target
        self.val_input = val_input
        self.val_past_cov = val_past_cov
        self.val_future_cov = val_future_cov
        self.seed = seed
    
    def optimize(self, model, n_calls=50):
        """
        Optimize hyperparameters using Bayesian optimization and Gaussian processes.

        Args:
            n_calls (int, optional): Number of calls to the objective function. Defaults to 50.
        
        Returns:
            skopt.OptimizeResult: Optimization result
        """
        assert model in ['nbeats', 'rf', 'xgb'], 'Model must be either nbeats, rf or xgb'
        if model == 'nbeats':
            hyperopt = HyperparameterOptimizationNBEATS(train_target=self.train_target, 
                                                        train_past_cov=self.train_past_cov, 
                                                        val_target=self.val_target, 
                                                        val_input=self.val_input, 
                                                        val_past_cov=self.val_past_cov, 
                                                        seed=self.seed)
        elif model == 'rf':
            hyperopt = HyperparameterOptimizationRandomForest(train_target=self.train_target, 
                                                              train_future_cov=self.train_future_cov, 
                                                              val_target=self.val_target,
                                                              val_input=self.val_input,
                                                              val_future_cov=self.val_future_cov,
                                                              seed=self.seed)
        else:
            hyperopt = HyperparameterOptimizationXGB(train_target=self.train_target, 
                                                     train_future_cov=self.train_future_cov, 
                                                     val_target=self.val_target,
                                                     val_input=self.val_input,
                                                     val_future_cov=self.val_future_cov,
                                                     seed=self.seed)
        
        result = hyperopt.optimize(n_calls=n_calls)
        
        if model == 'xgb':
            mapper =  { 'max_depth' : result.x[0],
                        'n_estimators' : result.x[1],
                        'n_jobs' : -1,
                        'lags' : int(result.x[3]),
                        'lags_future_covariates' : [k for k in range(result.x[4], 1)],
                        'learning_rate' : result.x[5],
                        'output_chunk_length' : hyperopt.output_length}
        
        elif model == 'rf':
            mapper = {  'max_depth': result.x[0],
                        'min_samples_split' : result.x[1],
                        'min_samples_leaf' : result.x[2],
                        'n_estimators' : result.x[3],
                        'n_jobs' : -1,
                        'lags' : int(result.x[5]),
                        'lags_future_covariates' : [k for k in range(result.x[6], 1)],
                        'output_chunk_length' : hyperopt.output_length}
        
        else:
            mapper = {'num_stacks' : int(result.x[0]),
                      'num_blocks' : int(result.x[1]), 
                      'layer_widths' : int(result.x[2]),
                      'n_epochs' : int(result.x[3]), 
                      'nr_epochs_val_period' : int(result.x[4]),
                      'input_chunk_length' : int(result.x[5]),
                      'output_chunk_length' : int(hyperopt.output_length),
                      'random_state' : int(self.seed),
                      'generic_architecture' : True,
                      'optimizer_kwargs' : {'lr' : 1e-3},
                      'likelihood' : QuantileRegression([0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]) }

        
        print(' -------------------------------------------------')
        print(f'| \t\t Model: {model}')
        print('|-------------------------------------------------')
        for dim, x in zip(hyperopt.space, result.x):

            print('| \t {} \t | \t {}'.format(round(x,3), dim.name))
            print(' -------------------------------------------------')
        
        return result, mapper

