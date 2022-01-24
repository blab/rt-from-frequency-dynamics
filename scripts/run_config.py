#!/usr/bin/env python
# coding: utf-8

import argparse
import numpyro
import pandas as pd
import yaml
import rt_from_frequency_dynamics as rf

def read_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(cf_data):
    # Load case data
    if cf_data["case_path"].endswith(".tsv"):
        raw_cases = pd.read_csv(cf_data["case_path"], sep="\t")
    else:
        raw_cases = pd.read_csv(cf_data["case_path"])
    
    # Load sequence count data
    if cf_data["seq_path"].endswith(".tsv"):
        raw_seq = pd.read_csv(cf_data["seq_path"], sep="\t")
    else:
        raw_seq = pd.read_csv(cf_data["seq_path"])
        
    # Load locations
    if "locations" in raw_cases:
        locations = cf["data"]["locations"]
    else: 
        # Check if raw_seq has location column
        locations = pd.unique(raw_seq["location"]) 
    
    return raw_cases, raw_seq, locations

def parse_cf(cf, var, dflt):
    if var in cf:
        return cf[var]
    else:
        return dflt

def parse_RLik(cf_m):
    # Don't like defaults being here...
    name = "GARW"
    gp = 0.5
    gdp = 10
    
     # Check for fields of interest 
    if "R_likelihood" in cf_m:
        name = cf_m["R_likelihood"]
    if "gam_prior" in cf_m:
        gp = cf_m["gam_prior"]
    if "gam_delta_prior" in cf_m:
        gdp = cf_m["gam_delta_prior"]
        
    # Select options
    if name == "GARW":
        return rf.GARW(gam_prior=gp, gam_delta_prior=gdp)
    elif name == "Free":
        return rf.FreeGrowth(gam_prior=gp)
    elif name == "Fixed":
        return rf.FixedGA(gam_prior=gp)
    
def parse_CLik(cf_m):
    # Don't like defaults being here...
    name = "ZINegBinom"
    pcd = 0.01
    
    # Check for fields of interest 
    if "C_likelihood" in cf_m:
        name = cf_m["C_likelihood"]
    if "prior_case_dispersion" in cf_m:
        pcd = cf_m["prior_case_dispersion"]
        
    # Select likelhood
    if name == "NegBinom":
        return rf.NegBinomCases(pcd)
    elif name == "ZINegBinom":
        return rf.ZINegBinomCases(pcd)
    elif name == "Poisson":
        return rf.PoisCases()
    elif name == "ZIPoisson":
        return rf.ZIPoisCases() 

def parse_SLik(cf_m):
    # Don't like defaults being here...
    name = "DirMultinomial"
    psd = 100.
    
    # Check for fields of interest 
    if "S_likelihood" in cf_m:
        name = cf_m["S_likelihood"]
    if "prior_seq_dispersion" in cf_m:
        psd = cf_m["prior_seq_dispersion"]
        
    if name == "DirMultinomial":
        return rf.DirMultinomialSeq(psd)
    elif name == "Multinomial":
        return rf.MultinomialSeq()

def parse_distributions(cf_dist):
    mn = cf_dist["mean"]
    sd = cf_dist["sd"]
    family = cf_dist["family"]
    if family == "LogNormal":
        return rf.discretise_lognorm(mn=mn, std=sd)
    elif family == "Gamma":
        return rf.discretise_gamma(mn=mn, std=sd)
    
def get_model(cf_m):
    # Processing hyperparameters
    seed_L = parse_cf(cf_m, "seed_L", dflt=7)
    forecast_L = parse_cf(cf_m, "forecast_L", dflt=0)
    k = parse_cf(cf_m, "k", dflt=10)
    
    # Processing generation time and delays
    gen = parse_distributions(cf_m["generation_time"])
    delays = [parse_distributions(d) for d in cf_m["delays"].values()]
    delays = rf.pad_delays(delays)
    
    # Processing likelihoods
    model = rf.RenewalModel(gen, delays, seed_L, forecast_L, k=k,
                       RLik = parse_RLik(cf_m), # Default is GARW
                       CLik = parse_CLik(cf_m), # Default is NegBinom
                       SLik = parse_SLik(cf_m)) # Default is DirMultinomial
    return model


def get_optim(cf_i):
    lr = float(parse_cf(cf_i, "lr", dflt=1e-2))
    iters = parse_cf(cf_i, "iters", dflt=50000)
    num_samples = parse_cf(cf_i, "num_samples", dflt=1500)
    return lr, iters, num_samples


def get_settings(cf_s):
    fit = parse_cf(cf_s, "fit", dflt=False)
    save = parse_cf(cf_s, "save", dflt=False)
    load = parse_cf(cf_s, "load", dflt=False)
    export_tsv = parse_cf(cf_s, "export_tsv", dflt=False)
    export_fig = parse_cf(cf_s, "export_fig", dflt=False)
    export_path = parse_cf(cf_s, "export_path", dflt=None)
    return fit, save, load, export_tsv, export_fig, export_path


def load_models(rc, rs, locations, RM, path=".", num_samples=1000):
    MP = rf.MultiPosterior()
    for i, loc in enumerate(locations):
        LD = rf.get_location_VariantData(rc, rs, loc)
        PH = rf.sample_loaded_posterior(LD, RM, num_samples=num_samples, path=path, name=loc)   
        MP.add_posterior(PH)
        print(f"Location {loc} finished {i+1} / {len(locations)}")
    return MP


def export_results(MP, ps, path_export):
    # Make directory 
    rf.make_model_directories(path_export)
    
    # Export model dataframes
    R_df = rf.gather_R(MP, ps)
    r_df = rf.gather_little_r(MP, ps)
    I_df = rf.gather_I(MP, ps)
    freq_df = rf.gather_freq(MP, ps)

    R_df.to_csv(f"{path_export}/Rt-combined.tsv", encoding='utf-8', sep='\t', index=False)
    r_df.to_csv(f"{path_export}/little-r-combined.tsv", encoding='utf-8', sep='\t', index=False)
    I_df.to_csv(f"{path_export}/I-combined.tsv", encoding='utf-8', sep='\t', index=False)
    freq_df.to_csv(f"{path_export}/freq-combined.tsv", encoding='utf-8', sep='\t', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Estimating variant growth rates."
    )
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()

    # Load configuration, data, and create model
    cf = read_config(args.config)
    raw_cases, raw_seq, locations = load_data(cf["data"])
    RM = get_model(cf["model"])
    lr, iters, num_samples = get_optim(cf["inference"])
    opt = numpyro.optim.Adam(step_size=lr)
    fit, save, load, export_tsv, export_fig, export_path = get_settings(cf["settings"])

    if export_path:
        rf.make_model_directories(export_path)
        
    if fit:
        MP = rf.fit_SVI_locations(raw_cases, raw_seq, locations, 
                                  RM, opt, 
                                  iters=iters, num_samples=num_samples, save=save, load=load, path=export_path)
    if load:
        MP = load_models(raw_cases, raw_seq, locations, RM, export_path, num_samples=num_samples)
        
    if export_tsv:
        ps = parse_cf(cf["settings"], "ps", dflt=[0.5, 0.8, 0.95])
        export_results(MP, ps, export_path)
