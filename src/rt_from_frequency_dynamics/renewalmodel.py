import jax.numpy as jnp

from .modeloptions import GARW, FixedGA, FreeGrowth
from .modelfactories  import _renewal_model_factory
from .Splines import Spline

# Abstract lineage model
class RenewalModel():
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
        self.model = _renewal_model_factory(
            self.g_rev, 
            self.delays, 
            self.seed_L, 
            self.forecast_L,
            self.RLik,
            self.CLik,
            self.SLik
        )

    def augment_data(self, data, order=4):
        T = len(data["cases"])
        s = jnp.linspace(0, T, self.k)
        data["X"] = Spline.matrix(jnp.arange(T), s, order=order)

class GARandomWalkModel(RenewalModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, GARW(), CLik, SLik)
        super().make_model()

class FreeGrowthModel(RenewalModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, FreeGrowth(), CLik, SLik)
        super().make_model()

class FixedGrowthModel(RenewalModel):
    def __init__(self, g, delays, seed_L, forecast_L, k=None, CLik=None, SLik=None):
        super().__init__(g, delays, seed_L, forecast_L, k, FixedGA(), CLik, SLik)
        super().make_model()
