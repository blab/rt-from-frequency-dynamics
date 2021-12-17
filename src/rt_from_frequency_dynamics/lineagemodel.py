import jax.numpy as jnp
from .modelfactories import _fixed_lineage_model_factory 
from .modelfactories import _free_lineage_model_factory
from .modelfactories import _decomp_lineage_model_factory
from .modelfactories import _GARW_model_factory

# Abstract lineage model
class LineageModel():
    def __init___(self):
        pass


class FixedGrowthModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L):
        self.g_rev = jnp.flip(g)
        self.delays = delays
        # self.rt_model = rt_model # TODO: Add R model interface
        # self.priors = priors
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        self.make_model()
    
    def make_model(self):
        # Model factory which inputs the desired thingy
        self.model = _fixed_lineage_model_factory(self.g_rev, self.delays, self.seed_L, self.forecast_L)
        # self.guide = _fixed_lineage_guide_factory(self.g_rev, self.delays, self.seed_L)


class FreeGrowthModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L):
        self.g_rev = jnp.flip(g)
        self.delays = delays
        # self.rt_model = rt_model
        # self.priors = priors
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        self.make_model()
    
    def make_model(self):
        self.model = _free_lineage_model_factory(self.g_rev, self.delays, self.seed_L, self.forecast_L)

class DecompGrowthModel(LineageModel):
    def __init__(self, g, delays, phi_0, seed_L, forecast_L):
        self.g_rev = jnp.flip(g)
        self.delays = delays
        self.phi_0 = phi_0
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        self.make_model()

    def make_model(self):
        self.model = _decomp_lineage_model_factory(self.g_rev, self.delays, self.phi_0, self.seed_L)

class GARandomWalkModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L):
        self.g_rev = jnp.flip(g)
        self.delays = delays
        # self.rt_model = rt_model
        # self.priors = priors
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        self.make_model()

    def make_model(self):
        self.model = _GARW_model_factory(self.g_rev, self.delays, self.seed_L, self.forecast_L)
