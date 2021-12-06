import numpyro
import pandas as pd

from rt_from_frequency_dynamics import get_standard_delays
from rt_from_frequency_dynamics import FreeGrowthModel, FixedGrowthModel

from rt_from_frequency_dynamics import fit_SVI_locations, gather_free_Rt, gather_fixed_Rt
from rt_from_frequency_dynamics import make_model_directories

if __name__ == "__main__":
    data_name = "variants-us"
    raw_cases = pd.read_csv(f"../data/{data_name}_location-case-counts.tsv", sep="\t")
    raw_seq = pd.read_csv(f"../data/{data_name}_location-variant-sequence-counts.tsv", sep="\t")

    locations = pd.unique(raw_seq["location"])

    # Defining Lineage Models
    seed_L = 7
    forecast_L = 0
    g, delays = get_standard_delays()
    
    LM_free = FreeGrowthModel(g, delays, seed_L, forecast_L)
    LM_fixed = FixedGrowthModel(g, delays, seed_L, forecast_L)

    # Params for fitting
    opt = numpyro.optim.Adam(step_size=1.0e-2)
    iters = 50_000
    num_samples = 3000
    save = True
    load = False

    path_base = f"../estimates/{data_name}"

    # Free Model settings
    path_free = path_base + "/free"
    make_model_directories(path_free)
    MP_free = fit_SVI_locations(raw_cases, raw_seq, locations, 
                                LM_free, opt, 
                                iters=iters, num_samples=num_samples, save=save, load=load, path=path_free)

    path_fixed = path_base + "/fixed"
    make_model_directories(path_fixed)
    MP_fixed = fit_SVI_locations(raw_cases, raw_seq, locations, 
                                 LM_fixed, opt, 
                                 iters=iters, num_samples=num_samples, save=save, load=load, path=path_fixed)

    # Saving samples from each state
    # save_posteriors(MP_free, path_free) # These files are Big!
    # save_posteriors(MP_fixed, path_fixed)

    # Exporting growth info
    ps = [0.95, 0.8, 0.5] # Which credible intevals to save
    R_free = gather_free_Rt(MP_free, ps, path=path_base, name=data_name)
    R_fixed, ga_fixed = gather_fixed_Rt(MP_fixed, ps, path=path_base, name=data_name)
