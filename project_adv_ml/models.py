from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression


class NBEATS(NBEATSModel):
    def __init__(self, input_chunk_length=10, output_chunk_length=1, n_epochs=100, stack_types=(NBEATSModel.TREND_STACK, NBEATSModel.SEASONALITY_STACK), nb_blocks_per_stack=3, nb_harmonics=None, learning_rate=3e-4, random_state=None):
        super().__init__(input_chunk_length=input_chunk_length,
                         output_chunk_length=output_chunk_length,
                         n_epochs=n_epochs,
                         stack_types=stack_types,
                         nb_blocks_per_stack=nb_blocks_per_stack,
                         nb_harmonics=nb_harmonics,
                         learning_rate=learning_rate,
                         random_state=random_state)
        

epochs = 200
blocks = 2*input_length # number of blocks in model        
layer_width = 32        # numer of weights in FC layer
batch = 2*input_length  # batch size
lr = 1e-3               # learning rate
val_wait = 1            # epochs to wait before evaluating the loss on the test/validation set
seed = 42               # random seed for regenerating results
n_samples = 100         # number of times a prediction is sampled from a probabilistic model
n_jobs = -1             # parallel processors to use;  -1 = all processors
split_proportion = 0.9  # train/test proportion

# quantiles for QuantileRegression argument
quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

# lower and upper quantiles for predictions
QL1, QL2 = 0.01, 0.05 
QU1, QU2 = 1 - QL1, 1 - QL2 
# labels for plotting
labelQ1 = f'{int(QU1 * 100)} / {int(QL1 * 100)} percentile band'
labelQ2 = f'{int(QU2 * 100)} / {int(QL2 * 100)} percentile band'


model_nbeats = NBEATSModel(input_chunk_length=input_length, 
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