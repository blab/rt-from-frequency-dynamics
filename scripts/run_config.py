#!/usr/bin/env python
# coding: utf-8

import argparse
import numpyro
import pandas as pd
import yaml
import rt_from_frequency_dynamics as rf


def parse_with_default(cf, var, dflt):
    if var in cf:
        return cf[var]
    else:
        print(f"Using default value for {var}")
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
    psd = 100.0

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


def parse_generation_time(cf_m):
    multiple_variants = "mean" not in cf_m["generation_time"]
    if multiple_variants:
        gen = rf.pad_delays(
            [
                parse_distributions(dist)
                for dist in cf_m["generation_time"].values()
            ]
        )
        v_names = [dist["name"] for dist in cf_m["generation_time"].values()]
    else:
        gen = parse_distributions(cf_m["generation_time"])
        v_names = None
    return gen, v_names


def parse_delays(cf_m):
    delays = rf.pad_delays(
        [parse_distributions(d) for d in cf_m["delays"].values()]
    )
    return delays


class RenewalConfig:
    def __init__(self, path):
        self.path = path
        self.config = self.read_config(path)

    def read_config(self, path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def load_data(self):
        data_cf = self.config["data"]

        # Load case data
        if data_cf["case_path"].endswith(".tsv"):
            raw_cases = pd.read_csv(data_cf["case_path"], sep="\t")
        else:
            raw_cases = pd.read_csv(data_cf["case_path"])

        # Load sequence count data
        if data_cf["seq_path"].endswith(".tsv"):
            raw_seq = pd.read_csv(data_cf["seq_path"], sep="\t")
        else:
            raw_seq = pd.read_csv(data_cf["seq_path"])

        # Load locations
        if "locations" in raw_cases:
            locations = data_cf["locations"]
        else:
            # Check if raw_seq has location column
            locations = pd.unique(raw_seq["location"])

        return raw_cases, raw_seq, locations

    def load_model(self):
        model_cf = self.config["model"]

        # Processing hyperparameters
        seed_L = parse_with_default(model_cf, "seed_L", dflt=7)
        forecast_L = parse_with_default(model_cf, "forecast_L", dflt=0)
        k = parse_with_default(model_cf, "k", dflt=10)

        # Processing generation time and delays
        gen, v_names = parse_generation_time(model_cf)
        delays = parse_delays(model_cf)

        # Processing likelihoods
        model = rf.RenewalModel(
            gen,
            delays,
            seed_L,
            forecast_L,
            k=k,
            RLik=parse_RLik(model_cf),  # Default is GARW
            CLik=parse_CLik(model_cf),  # Default is NegBinom
            SLik=parse_SLik(model_cf),
            v_names=v_names,
        )  # Default is DirMultinomial
        return model

    def load_optim(self):
        infer_cf = self.config["inference"]
        lr = float(parse_with_default(infer_cf, "lr", dflt=1e-2))
        iters = int(parse_with_default(infer_cf, "iters", dflt=50000))
        num_samples = int(
            parse_with_default(infer_cf, "num_samples", dflt=1500)
        )
        return lr, iters, num_samples

    def load_settings(self):
        settings_cf = self.config["settings"]
        fit = parse_with_default(settings_cf, "fit", dflt=False)
        save = parse_with_default(settings_cf, "save", dflt=False)
        load = parse_with_default(settings_cf, "load", dflt=False)
        export_tsv = parse_with_default(settings_cf, "export_tsv", dflt=False)
        export_path = parse_with_default(settings_cf, "export_path", dflt=None)
        return fit, save, load, export_tsv, export_path


def load_models(rc, rs, locations, RM, path=None, num_samples=1000):
    if path is None:
        path = "."
    MP = rf.MultiPosterior()
    for i, loc in enumerate(locations):
        LD = rf.get_location_VariantData(rc, rs, loc)
        PH = rf.sample_loaded_posterior(
            LD, RM, num_samples=num_samples, path=path, name=loc
        )
        MP.add_posterior(PH)
        print(f"Location {loc} finished {i+1} / {len(locations)}")
    return MP


def export_results(MP, ps, path_export, data_name):
    # Make directory
    rf.make_model_directories(path_export)

    # Export model dataframes
    R_df = rf.gather_R(MP, ps)
    r_df = rf.gather_little_r(MP, ps)
    I_df = rf.gather_I(MP, ps)
    freq_df = rf.gather_freq(MP, ps)

    R_df.to_csv(
        f"{path_export}/{data_name}_Rt-combined.tsv",
        encoding="utf-8",
        sep="\t",
        index=False,
    )
    r_df.to_csv(
        f"{path_export}/{data_name}_little-r-combined.tsv",
        encoding="utf-8",
        sep="\t",
        index=False,
    )
    I_df.to_csv(
        f"{path_export}/{data_name}_I-combined.tsv",
        encoding="utf-8",
        sep="\t",
        index=False,
    )
    freq_df.to_csv(
        f"{path_export}/{data_name}_freq-combined.tsv",
        encoding="utf-8",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Estimating variant growth rates."
    )
    parser.add_argument("--config", help="path to config file")
    args = parser.parse_args()

    # Load configuration, data, and create model
    config = RenewalConfig(args.config)
    print(f"Config loaded: {config.path}")

    raw_cases, raw_seq, locations = config.load_data()
    print("Data loaded sucessfuly")

    renewal_model = config.load_model()
    print("Model created.")

    lr, iters, num_samples = config.load_optim()
    opt = numpyro.optim.Adam(step_size=lr)
    print("Optimizer defined.")

    fit, save, load, export_tsv, export_path = config.load_settings()
    print("Settings loaded")

    # Find export path
    if export_path:
        rf.make_model_directories(export_path)

    # Fit or load model results
    if fit:
        print("Fitting model")
        multi_posterior = rf.fit_SVI_locations(
            raw_cases,
            raw_seq,
            locations,
            renewal_model,
            opt,
            iters=iters,
            num_samples=num_samples,
            save=save,
            load=load,
            path=export_path,
        )
    elif load:
        print("Loading results")
        multi_posterior = load_models(
            raw_cases,
            raw_seq,
            locations,
            renewal_model,
            export_path,
            num_samples=num_samples,
        )
    else:
        print("No models fit or results loaded.")
        MP = None

    # Export results
    if export_tsv and (fit or load):
        print(f"Exporting results as .tsv at {export_path}")
        ps = parse_with_default(
            config.config["settings"], "ps", dflt=[0.5, 0.8, 0.95]
        )
        data_name = config.config["data"]["name"]
        export_results(multi_posterior, ps, export_path, data_name)
