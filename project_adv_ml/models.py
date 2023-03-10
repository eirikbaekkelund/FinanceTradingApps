from darts.models import NBEATSModel, XGBModel, LinearRegressionModel, BlockRNNModel

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
        
    def fit(self, target, past_covs, mc_dropout=True):
        return self.fit(series=target,
                        past_covariates=past_covs,
                        n_jobs=-1,
                        mc_dropout=mc_dropout)

class TraditionalNBeats(NBEATSModel):
    def __init__(self, input_chunk_length=10, output_chunk_length=1, n_epochs=100, nb_blocks_per_stack=3, nb_harmonics=None, learning_rate=3e-4, random_state=None):
        super().__init__(input_chunk_length=input_chunk_length,
                         output_chunk_length=output_chunk_length,
                         n_epochs=n_epochs,
                         nb_blocks_per_stack=nb_blocks_per_stack,
                         nb_harmonics=nb_harmonics,
                         learning_rate=learning_rate,
                         random_state=random_state)
        
    def fit(self, target, past_covs, mc_dropout=True):
        return self.fit(series=target,
                        past_covariates=past_covs,
                        n_jobs=-1,
                        mc_dropout=mc_dropout)

