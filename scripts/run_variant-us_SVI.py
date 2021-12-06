import numpyro
import pandas as pd

from rt_from_frequency_dynamics import get_standard_delays
from rt_from_frequency_dynamics import FreeGrowthModel, FixedGrowthModel

from rt_from_frequency_dynamics import fit_SVI_locations, save_posteriors, gather_growth_info

if __name__ == "__main__":
    raw_cases = pd.read_csv("../data/variants-us_location-case-counts.tsv", sep="\t")
    raw_seq = pd.read_csv("../data/variants-us_location-variant-sequence-counts.tsv", sep="\t")

    locations = pd.unique(raw_seq["location"])
    # locations = ["Michigan"]

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

    # Free Model settings
    path_free = "../results/variants-us/free"
    MP_free = fit_SVI_locations(raw_cases, raw_seq, locations, 
                                LM_free, opt, 
                                iters=iters, num_samples=num_samples, save=save, load=load, path=path_free)

    path_fixed = "../results/variants-us/fixed"
    MP_fixed = fit_SVI_locations(raw_cases, raw_seq, locations, 
                                 LM_fixed, opt, 
                                 iters=iters, num_samples=num_samples, save=save, load=load, path=path_fixed)

    # Saving samples from each state
    # save_posteriors(MP_free, path_free) # These files are Big!
    # save_posteriors(MP_fixed, path_fixed)

    # Exporting growth info
    ps = [0.95, 0.8, 0.5] # Which credible intevals to save
    R_free, _ = gather_growth_info(MP_free, ps, ga=False, path=path_free)
    R_fixed, ga_fixed = gather_growth_info(MP_fixed, ps, ga=True, path=path_fixed)
