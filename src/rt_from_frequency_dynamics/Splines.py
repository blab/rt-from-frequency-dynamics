from jax import vmap
import jax.numpy as jnp

class Spline:
    @staticmethod
    def _omega(s1, s2, t):
        return jnp.where(s1==s2, jnp.zeros_like(t), (t-s1)/(s2-s1))

    @staticmethod
    def _basis(t, s, order, i):
        if order == 1:
            return jnp.where(
                (t>=s[i])*(t<s[i+1]), 
                jnp.ones_like(t),
                jnp.zeros_like(t))
        
        # Recurse left
        w1 = Spline._omega(s[i], s[i+order-1], t)
        B1 = Spline._basis(t, s, order-1, i)
        
        # Recurse right
        w2 = Spline._omega(s[i+1], s[i+order], t)
        B2 = Spline._basis(t, s, order-1, i+1)
        return w1*B1 + (1-w2)*B2

    @staticmethod
    def matrix(t, s, order):
        _s = jnp.pad(s, mode="edge", pad_width=(order-1)) # Extend knots
        _sb = lambda i: Spline._basis(t,_s, order, i)
        X = vmap(_sb)(jnp.arange(0,len(s)+order-2)) # Make spline basis
        return X.T

class SplineDeriv:
    @staticmethod
    def _omegap(s1, s2, t):
        return jnp.where(s1 == s2, jnp.zeros_like(t), jnp.reciprocal(s2-s1) )

    @staticmethod
    def _basis(t, s, order, i):
        if order == 1:
            return jnp.where(
                (t>=s[i])*(t<s[i+1]), 
                jnp.ones_like(t),
                jnp.zeros_like(t))
        
        # Recurse left
        w1 = SplineDeriv._omegap(s[i], s[i+order-1], t)
        B1 = Spline._basis(t, s, order-1, i)
        
        # Recurse right
        w2 = SplineDeriv._omegap(s[i+1], s[i+order], t)
        B2 = Spline._basis(t, s, order-1, i+1)
        return (order-1) * (w1*B1 - w2*B2)

    @staticmethod
    def matrix(t, s, order):
        _s = jnp.pad(s, mode="edge", pad_width=(order-1)) # Extend knots
        _sb = lambda i: SplineDeriv._basis(t, _s, order, i) 
        X = vmap(_sb)(jnp.arange(0,len(s)+order-2)) # Make spline basis
        return X.T
