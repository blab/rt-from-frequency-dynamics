import numpyro
from numpyro.infer.autoguide import AutoMultivariateNormal
import pandas as pd
from src.rt_from_frequency_dynamics import *

ps = DefaultAes.ps

def run_SVI(LD, optimizer, num_samples=1000, iters=100_000, name="test", path=".", save=True, export=True, load=False):       
    X = make_breakpoint_splines(LD.cases, 20)
    data = LD.make_numpyro_input(X)

    # Defining model
    g, delays = get_standard_delays()
    LM = FixedGrowthModel(g, delays, 7, 0)

    # Run SVI
    SVIH = SVIHandler(optimizer=optimizer)
    guide = AutoMultivariateNormal(LM.model)
    
    if load:
        SVIH.load_state(f"{path}/models/{name}_svi.p")
    
    loss = SVIH.fit(LM.model, guide, data, iters, log_each=0)
    print(f"Model {name} finished. Final loss: {loss}")
    
    if jnp.isnan(loss):
        return False
    if save:
        SVIH.save_state(f"{path}/models/{name}_svi.p")
    
    # Get samples
    samples = SVIH.predict(LM.model, guide, data, num_samples=num_samples)
    dataset = to_arviz(samples)
    
    if export:
        # Get dataframes
        R_dataframe = pd.DataFrame(get_R(dataset, LD, ps[:-1], name))
        ga_dataframe = pd.DataFrame(get_growth_advantage(dataset, LD, ps[:-1], name))
    
        R_dataframe.to_csv(f"{path}/Rt/Rt_{name}.csv", encoding='utf-8', index=False)
        ga_dataframe.to_csv(f"{path}/ga/ga_{name}.csv", encoding='utf-8', index=False)
    return True

def get_state_LD(rc, rs, loc):
    rc_l = rc[rc.location == loc].copy()
    rs_l = rs[rs.location==loc].copy()
    return LineageData(rc_l, rs_l)

def run_locations(rc, rs, locations, optimizer, **kwargs):
    n_locations = len(locations)
    sucesses = []
    for i, loc in enumerate(locations):
        LD = get_state_LD(raw_cases, raw_seq, loc)
        model_name = loc.replace(" ", "_")
        sucess = run_SVI(LD, optimizer, name=model_name, **kwargs)
        sucesses.append(sucess)
        print(f'Location {loc} finished ({i+1}/{n_locations}).')
    return locations[successes]

def combine_exports(rc, rs, locations, path):   
    rt_list = []
    ga_list = []
    for i, loc in enumerate(locations):
        _loc = loc.replace(" ", "_")
        rt_list.append(pd.read_csv(f"{path}/Rt/Rt_{_loc}.csv"))
        ga_list.append(pd.read_csv(f"{path}/ga/ga_{_loc}.csv"))
    return pd.concat(rt_list), pd.concat(ga_list)


if __name__ == "__main__":
    raw_cases = pd.read_csv("../../rt-from-frequency-dynamics/data/location-case-counts.tsv", sep="\t")
    raw_seq = pd.read_csv("../../rt-from-frequency-dynamics/data/location-variant-sequence-counts.tsv", sep="\t")

    locations = pd.unique(raw_seq.location)

    optimizer = numpyro.optim.Adam(step_size=1.5e-4)
    num_samples = 1
    iters = 200_000
    path = "./sims/all-states-preprint"
    save = True
    export = True
    load = False
    succeded = run_locations(raw_cases,
                             raw_seq,
                             locations,
                             optimizer,
                             num_samples=num_samples,
                             iters=iters,
                             path=path,
                             save=save,
                             export=export,
                             load=load)
