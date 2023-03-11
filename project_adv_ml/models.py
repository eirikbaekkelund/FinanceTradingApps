from darts.models import NBEATSModel
from darts.metrics import rmse
from darts.utils.likelihood_models import QuantileRegression
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

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
    def __init__(self, input_length, output_length, train_target, train_past_cov, val_target, val_input, val_past_cov, seed=42):
        assert isinstance(input_length, int), "input_length must be of type int"
        assert isinstance(output_length, int), "output_length must be of type int"
        assert isinstance(seed, int), "seed must be of type int"

        self.input_length = input_length
        self.output_length = output_length
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
            Integer(50, 100, name='n_epochs'),
            Integer(1, 3, name='nr_epochs_val_period')]

    # insert static method if 'self' is not used
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
        n_stacks, n_blocks, layer_width, epochs, val_wait = map(int, params)
        quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        model = NBEATSModel(input_chunk_length=self.input_length, 
                            output_chunk_length=self.output_length,
                            num_stacks=n_stacks,
                            num_blocks=n_blocks,
                            layer_widths=layer_width,
                            n_epochs=epochs,
                            likelihood=QuantileRegression(quantiles),
                            optimizer_kwargs={"lr": 1e-3},
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
        
        def objective(num_stacks, num_blocks, layer_width, n_epochs, nr_epochs_val_period):
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
            params = (num_stacks, num_blocks, layer_width, n_epochs, nr_epochs_val_period)
            return self.objective_nbeats(params)
        
        result = gp_minimize(objective, self.space, n_calls=n_calls)
        return result
