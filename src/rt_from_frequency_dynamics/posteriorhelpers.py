import arviz as az
import jax.numpy as jnp
import numpy as np


def reshape_for_arviz(samples):
    new_samples = dict()
    for key, _ in samples.items():
        new_samples[key] = jnp.expand_dims(samples[key], 0)
    return new_samples


def to_arviz(samples):
    dataset = az.convert_to_inference_data(reshape_for_arviz(samples))
    return dataset

def get_growth_advantage(dataset, LD, ps, name):
    medians = dataset.posterior["ga"].median(dim="draw").values[0]
    N_variant = medians.shape[0]
    seq_names = LD.seq_names

    ga = []
    for i in range(len(ps)):
        ga.append(jnp.array(az.hdi(dataset, var_names="ga", hdi_prob=ps[i])["ga"]))

    v_dict = dict()
    v_dict["location"] = []
    v_dict["variant"] = []
    v_dict["median_ga"] = []
    for p in ps:
        v_dict[f"ga_upper_{round(p * 100)}"] = []
        v_dict[f"ga_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        v_dict["location"].append(name)
        v_dict["variant"].append(seq_names[variant])
        v_dict["median_ga"].append(medians[variant])
        for i,p in enumerate(ps):
            v_dict[f"ga_upper_{round(p * 100)}"].append(ga[i][variant, 1])
            v_dict[f"ga_lower_{round(p * 100)}"].append(ga[i][variant, 0])
        
    return v_dict

def get_R(dataset, LD, ps, name):
    R_medians = dataset.posterior["R"].median(dim="draw").values[0]
    freq_medians = dataset.posterior["freq"].median(dim="draw").values[0]
    N_variant = R_medians.shape[1]
    T = R_medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates

    R = []
    for i in range(len(ps)):
        R.append(jnp.array(az.hdi(dataset, var_names="R", hdi_prob=ps[i])["R"]))

    R_dict = dict()
    R_dict["date"] = []
    R_dict["location"] = []
    R_dict["variant"] = []
    R_dict["median_R"] = []
    R_dict["median_freq"] = []
    
    for p in ps:
        R_dict[f"R_upper_{round(p * 100)}"] = []
        R_dict[f"R_lower_{round(p * 100)}"] = []
        
    for variant in range(N_variant):
        R_dict["date"] += list(dates)
        R_dict["location"] += [name] * T
        R_dict["variant"] += [seq_names[variant]] * T
        R_dict["median_R"] += list(R_medians[:, variant])
        R_dict["median_freq"] += list(freq_medians[:, variant])
        for i,p in enumerate(ps):
            R_dict[f"R_upper_{round(ps[i] * 100)}"] += list(R[i][:, variant, 1])
            R_dict[f"R_lower_{round(ps[i] * 100)}"] += list(R[i][:, variant, 0])

    return R_dict

def get_little_r(dataset, g, LD, ps, name):
    # Get generation time
    mn = np.sum([p * (x+1) for x, p in enumerate(g)]) # Get mean of discretized generation time
    sd = np.sqrt(np.sum([p * (x+1) **2 for x, p in enumerate(g)])-mn**2) # Get sd of discretized generation time
    e_ = sd**2 / mn**2
    l = mn / (sd**2)
    
    # Set up conversion
    def _to_little_r(R):
        return (jnp.float_power(R, e_) - 1) * l
    
    R_medians = dataset.posterior["R"].median(dim="draw").values[0]
    freq_medians = dataset.posterior["freq"].median(dim="draw").values[0]
    N_variant = R_medians.shape[1]
    T = R_medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates
    
    R = []
    for i in range(len(ps)):
        R.append(jnp.array(az.hdi(dataset, var_names="R", hdi_prob=ps[i])["R"]))

    r_dict = dict()
    r_dict["date"] = []
    r_dict["location"] = []
    r_dict["variant"] = []
    r_dict["median_r"] = []
    r_dict["median_freq"] = []
    
    for p in ps:
        r_dict[f"r_upper_{round(p * 100)}"] = []
        r_dict[f"r_lower_{round(p * 100)}"] = []
        
    for variant in range(N_variant):
        r_dict["date"] += list(dates)
        r_dict["location"] += [name] * T
        r_dict["variant"] += [seq_names[variant]] * T
        r_dict["median_r"] += list(_to_little_r(R_medians[:, variant]))
        r_dict["median_freq"] += list(freq_medians[:, variant])
        for i,p in enumerate(ps):
            r_dict[f"r_upper_{round(ps[i] * 100)}"] += list(_to_little_r(R[i][:, variant, 1]))
            r_dict[f"r_lower_{round(ps[i] * 100)}"] += list(_to_little_r(R[i][:, variant, 0]))

    return r_dict
