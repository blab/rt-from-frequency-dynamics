import jax.numpy as jnp

from rt_from_frequency_dynamics.modelhelpers import make_breakpoint_splines
from .modeloptions import GARW, FixedGA, FreeGrowth
from .modelfactories  import _model_factory

# Abstract lineage model
class LineageModel():
    def __init__(self, g, delays, seed_L, forecast_L, 
                 k=None,
                 RLik=None, 
                 CLik=None, 
                 SLik=None):
        self.g_rev = jnp.flip(g)
        self.delays = delays
        self.seed_L = seed_L
        self.forecast_L = forecast_L

        if k is None:
            k = 20
        self.k = k

        self.RLik = RLik
        self.CLik = CLik
        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model = _model_factory(
            self.g_rev, 
            self.delays, 
            self.seed_L, 
            self.forecast_L,
            self.RLik,
            self.CLik,
            self.SLik
        )

    def augment_data(self, data):
        data["X"] = make_breakpoint_splines(len(data["cases"]), self.k)

class GARandomWalkModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, GARW(), CLik, SLik)
        super().make_model()

class FreeGrowthModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, FreeGrowth(), CLik, SLik)
        super().make_model()

class FixedGrowthModel(LineageModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, FixedGA(), CLik, SLik)
        super().make_model()
