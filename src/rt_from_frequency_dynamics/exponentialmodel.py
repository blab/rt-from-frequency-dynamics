import jax.numpy as jnp

from .modelfactories import _exp_model_factory
from .Splines import Spline, SplineDeriv


class ExpModel:
    def __init__(self, g=None, delays=None, k=None, CLik=None, SLik=None):
        if g is not None:
            self.g_rev = jnp.flip(g)
        if delays is not None:
            self.delays = delays
        if k is None:
            k = 20
        self.k = k

        self.CLik = CLik
        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model = _exp_model_factory(
            g_rev=self.g_rev, delays=self.delays, CaseLik=self.CLik, SeqLik=self.SLik
        )

    def augment_data(self, data, order=4):
        T = len(data["cases"])
        s = jnp.linspace(0, T, self.k)
        data["X"] = Spline.matrix(jnp.arange(T), s, order=order)
        data["X_prime"] = SplineDeriv.matrix(jnp.arange(T), s, order=order)
