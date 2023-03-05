from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression


class SeasonalNBEATS(NBEATSModel):
    def __init__(self, input_chunk_length=10, output_chunk_length=1, n_epochs=100, stack_types=(NBEATSModel.TREND_STACK, NBEATSModel.SEASONALITY_STACK), nb_blocks_per_stack=3, nb_harmonics=None, learning_rate=3e-4, random_state=None):
        super().__init__(input_chunk_length=input_chunk_length,
                         output_chunk_length=output_chunk_length,
                         n_epochs=n_epochs,
                         stack_types=stack_types,
                         nb_blocks_per_stack=nb_blocks_per_stack,
                         nb_harmonics=nb_harmonics,
                         learning_rate=learning_rate,
                         random_state=random_state)
        


model = NBEATSModel(input_chunk_length=input_length, 
                     output_chunk_length=output_length,
                     num_stacks=blocks,
                     layer_widths=layer_width,
                     n_epochs=epochs,
                     likelihood = QuantileRegression(quantiles),
                     optimizer_kwargs = {"lr" : 1e-3},
                     generic_architecture=True,
                     trend_polynomial_degree=2,
                     random_state=seed,
                     nr_epochs_val_period=val_wait,
                     #pl_trainer_kwargs = {"accelerator": "cpu", "devices": 8 } 
                     )
predictor_nbeats = model_nbeats.fit(series=series_train, 
                       past_covariates=past_covariates,               
                       verbose=False,
                       epochs=epochs)