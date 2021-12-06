import pandas as pd

from numpyro.infer.autoguide import AutoMulitvariateNormal

from rt_from_frequency_dynamics import LineageData
from rt_from_frequency_dynamics import get_R, get_growth_advantage
from rt_from_frequency_dynamics import SVIHandler, PosteriorHandler, MultiPosterior

def get_location_LineageData(rc, rs, loc):
    rc_l = rc[rc.location == loc].copy()
    rs_l = rs[rs.location==loc].copy()
    return LineageData(rc_l, rs_l)

def fit_SVI(LD, LM, opt, iters=100_000, num_samples = 1000, path = ".", name="Test", load=False, save=False):
    # Defining optimization
    SVIH = SVIHandler(optimizer=opt)
    guide = AutoMulitvariateNormal(LM.model)

    # Upacking data
    data = LD.make_numpyro_input()
    # Loading past state

    if load:
        file_name = name.replace(" ", "-")
        SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    # Fitting model 
    if iters > 0:
        loss = SVIH.fit(LM.model, guide, data, iters, log_each=0)
    
    if save:
        file_name = name.replace(" ", "-")
        SVIH.save_state(f"{path}/models/{name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples = num_samples)
    return PosteriorHandler(dataset=dataset,LD=LD, name=name)  

def fit_SVI_locations(rc, rs, locations, LM, opt, **fit_kwargs):
    n_locations = len(locations)
    MP = MultiPosterior()
    for i, loc in enumerate(locations):
        LD = get_location_LineageData(rc, rs, loc)
        PH = fit_SVI(LD, LM, opt, **fit_kwargs)
        MP.add_posterior(PH)
        print(f'Location {loc} finished ({i+1}/{n_locations}).')
    return MP

def save_posteriors(MP, path):
    for name, p in MP.locator.items():
        p.save_posterior(f"{path}/{name}_samples.json")
    return None

def gather_growth_info(MP, ps, ga=False):
    R_dfs = []
    for name, p in MP.locator.items():
        R_dfs.append(pd.DataFrame(get_R(p.dataset, p.LD, ps, name)))

    if not ga:
        return pd.concat(R_dfs), None

    ga_dfs = []
    for name, p in MP.locator.items():
        ga_dfs.append(pd.DataFrame(get_growth_advantage(p.dataset, p.LD, ps, name)))

    return pd.concat(R_dfs), pd.concat(ga_dfs)
