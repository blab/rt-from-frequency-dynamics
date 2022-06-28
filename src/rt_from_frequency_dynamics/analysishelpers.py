import pandas as pd

import jax.numpy as jnp
import numpyro
from numpyro.infer.autoguide import AutoMultivariateNormal

from rt_from_frequency_dynamics import VariantData
from rt_from_frequency_dynamics import (
    get_R,
    get_growth_advantage,
    get_growth_advantage_time,
    get_little_r,
    get_I,
    get_freq,
)
from .inferencehandler import SVIHandler, MCMCHandler
from .posteriorhandler import PosteriorHandler, MultiPosterior


def get_location_VariantData(rc, rs, loc):
    rc_l = rc[rc.location == loc].copy()
    rs_l = rs[rs.location == loc].copy()
    return VariantData(rc_l, rs_l)


def fit_SVI(
    VD,
    LM,
    opt,
    iters=100_000,
    num_samples=1000,
    path=".",
    name="Test",
    load=False,
    save=False,
):
    # Defining optimization
    SVIH = SVIHandler(optimizer=opt)
    guide = AutoMultivariateNormal(LM.model)

    # Upacking data
    data = VD.make_numpyro_input()
    LM.augment_data(data)
    # Loading past state
    if load:
        file_name = name.replace(" ", "-")
        SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    # Fitting model
    if iters > 0:
        SVIH.fit(LM.model, guide, data, iters)

    if save:
        file_name = name.replace(" ", "-")
        SVIH.save_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data=data, num_samples=num_samples)

    # Adding in losses
    dataset["loss"] = SVIH.losses
    return PosteriorHandler(dataset=dataset, data=VD, name=name)


def fit_MCMC(
    VD,
    LM,
    kernel=None,
    num_samples=1000,
    num_warmup=500,
    path=".",
    name="Test",
    load=False,
    save=False,
):
    # Defining MCMC algorithm
    MC = MCMCHandler(kernel=kernel)
    # Unpacking data
    data = VD.make_numpyro_input()
    LM.augment_data(data)
    MC.fit(LM.model, data, num_warmup, num_samples)
    dataset = MC.predict(LM.model, data=data, num_samples=num_samples)
    return PosteriorHandler(dataset=dataset, data=VD, name=name)


def fit_SVI_locations(rc, rs, locations, LM, opt, **fit_kwargs):
    n_locations = len(locations)
    MP = MultiPosterior()
    for i, loc in enumerate(locations):
        VD = get_location_VariantData(rc, rs, loc)
        PH = fit_SVI(VD, LM, opt, name=loc, **fit_kwargs)
        MP.add_posterior(PH)
        print(f"Location {loc} finished ({i+1}/{n_locations}).")
    return MP


def fit_MCMC_locations(rc, rs, locations, LM, **fit_kwargs):
    n_locations = len(locations)
    MP = MultiPosterior()
    for i, loc in enumerate(locations):
        VD = get_location_VariantData(rc, rs, loc)
        PH = fit_MCMC(VD, LM, name=loc, **fit_kwargs)
        MP.add_posterior(PH)
        print(f"Location {loc} finished ({i+1}/{n_locations}).")
    return MP


def save_posteriors(MP, path):
    for name, p in MP.locator.items():
        p.save_posterior(f"{path}/posteriors/{name}_samples.json")
    return None


def sample_loaded_posterior(VD, LM, num_samples=1000, path=".", name="Test"):
    # Defining optimization
    SVIH = SVIHandler(optimizer=numpyro.optim.Adam(step_size=1e-2))
    guide = AutoMultivariateNormal(LM.model)

    # Upacking data
    data = VD.make_numpyro_input()
    LM.augment_data(data)

    # Loading past state
    file_name = name.replace(" ", "-")
    SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples=num_samples)
    return PosteriorHandler(dataset=dataset, data=VD, name=name)


def unpack_model(MP, name):
    posterior = MP.get(name)
    return posterior.dataset, posterior.data


def gather_R(MP, ps, forecast=False):
    R_dfs = []
    for name, p in MP.locator.items():
        R_dfs.append(
            pd.DataFrame(get_R(p.dataset, p.data, ps, name, forecast=forecast))
        )
    return pd.concat(R_dfs)


def gather_little_r(MP, ps, forecast=False):
    r_dfs = []
    for name, p in MP.locator.items():
        r_dfs.append(
            pd.DataFrame(
                get_little_r(p.dataset, p.data, ps, name, forecast=forecast)
            )
        )
    return pd.concat(r_dfs)


def gather_ga(MP, ps, rel_to="other"):
    ga_dfs = []
    for name, p in MP.locator.items():
        ga_dfs.append(
            pd.DataFrame(
                get_growth_advantage(
                    p.dataset, p.data, ps, name, rel_to=rel_to
                )
            )
        )
    return pd.concat(ga_dfs)


def gather_ga_time(MP, ps, rel_to="other"):
    ga_dfs = []
    for name, p in MP.locator.items():
        ga_dfs.append(
            pd.DataFrame(
                get_growth_advantage_time(
                    p.dataset, p.data, ps, name, rel_to=rel_to
                )
            )
        )
    return pd.concat(ga_dfs)


def gather_I(MP, ps, forecast=False):
    I_dfs = []
    for name, p in MP.locator.items():
        I_dfs.append(
            pd.DataFrame(get_I(p.dataset, p.data, ps, name, forecast=forecast))
        )
    return pd.concat(I_dfs)


def gather_freq(MP, ps, forecast=False):
    freq_dfs = []
    for name, p in MP.locator.items():
        freq_dfs.append(
            pd.DataFrame(
                get_freq(p.dataset, p.data, ps, name, forecast=forecast)
            )
        )
    return pd.concat(freq_dfs)
