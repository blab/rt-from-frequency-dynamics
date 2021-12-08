import pandas as pd

import numpyro
from numpyro.infer.autoguide import AutoMultivariateNormal

from rt_from_frequency_dynamics import LineageData
from rt_from_frequency_dynamics import get_R, get_growth_advantage, get_little_r
from rt_from_frequency_dynamics import SVIHandler, PosteriorHandler, MultiPosterior

def get_location_LineageData(rc, rs, loc):
    rc_l = rc[rc.location == loc].copy()
    rs_l = rs[rs.location==loc].copy()
    return LineageData(rc_l, rs_l)

def fit_SVI(LD, LM, opt, iters=100_000, num_samples = 1000, path = ".", name="Test", load=False, save=False):
    # Defining optimization
    SVIH = SVIHandler(optimizer=opt)
    guide = AutoMultivariateNormal(LM.model)

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
        SVIH.save_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples = num_samples)
    return PosteriorHandler(dataset=dataset,LD=LD, name=name)  

def fit_SVI_locations(rc, rs, locations, LM, opt, **fit_kwargs):
    n_locations = len(locations)
    MP = MultiPosterior()
    for i, loc in enumerate(locations):
        LD = get_location_LineageData(rc, rs, loc)
        PH = fit_SVI(LD, LM, opt, name=loc, **fit_kwargs)
        MP.add_posterior(PH)
        print(f'Location {loc} finished ({i+1}/{n_locations}).')
    return MP

def save_posteriors(MP, path):
    for name, p in MP.locator.items():
        p.save_posterior(f"{path}/posteriors/{name}_samples.json")
    return None

def gather_growth_info(MP, ps, ga=False, g=None):
    R_dfs = []
    for name, p in MP.locator.items():
        R_dfs.append(pd.DataFrame(get_R(p.dataset, p.LD, ps, name)))

    R_df = pd.concat(R_dfs)
    
    r_df = None
    if g is not None: # If generation time provided, return little r
        r_dfs = []
        for name, p in MP.locator.items():
            r_dfs.append(pd.DataFrame(get_little_r(p.dataset, g, p.LD, ps, name)))
        r_df = pd.concat(r_dfs)

    ga_df = None
    if ga: # If growth advantage wanted, get it
        ga_dfs = []
        for name, p in MP.locator.items():
            ga_dfs.append(pd.DataFrame(get_growth_advantage(p.dataset, p.LD, ps, name)))

        ga_df = pd.concat(ga_dfs)
    return R_df, r_df, ga_df

def gather_free_Rt(MP, ps, g=None, path=".", name=""):
    R_df, r_df, _ = gather_growth_info(MP, ps, ga=False, g=g)
    R_df.to_csv(f"{path}/{name}_Rt-combined-free.tsv", encoding='utf-8', sep='\t', index=False)
    if r_df is not None:
        r_df.to_csv(f"{path}/{name}_little-r-combined-free.tsv", encoding='utf-8', sep='\t', index=False)
    return R_df, r_df

def gather_fixed_Rt(MP, ps, g=None, path=".", name=""):
    R_df, r_df, ga_df = gather_growth_info(MP, ps, ga=True, g=g) 
    R_df.to_csv(f"{path}/{name}_Rt-combined-fixed.tsv", encoding='utf-8', sep='\t', index=False)
    ga_df.to_csv(f"{path}/{name}_ga-combined-fixed.tsv", encoding='utf-8', sep='\t', index=False)
    if r_df is not None:
        r_df.to_csv(f"{path}/{name}_little-r-combined-fixed.tsv", encoding='utf-8', sep='\t', index=False)

    return R_df, r_df, ga_df

def sample_loaded_posterior(LD, LM, num_samples = 1000, path = ".", name="Test"):
    # Defining optimization
    SVIH = SVIHandler(optimizer=numpyro.optim.Adam(step_size=1e-2))
    guide = AutoMultivariateNormal(LM.model)

    # Upacking data
    data = LD.make_numpyro_input()
    
    # Loading past state
    file_name = name.replace(" ", "-")
    SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples = num_samples)
    return PosteriorHandler(dataset=dataset, LD=LD, name=name)

def unpack_model(MP, name):
    posterior = MP.get(name)
    return posterior.dataset, posterior.LD
