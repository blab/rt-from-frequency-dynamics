import arviz as az
import jax.numpy as jnp
import numpy as np

from .datahelpers import forecast_dates

def reshape_for_arviz(samples):
    new_samples = dict()
    for key, _ in samples.items():
        new_samples[key] = jnp.expand_dims(samples[key], 0)
    return new_samples

def to_arviz(samples):
    dataset = az.convert_to_inference_data(reshape_for_arviz(samples))
    return dataset

def get_growth_advantage(dataset, LD, ps, name, rel_to="other"):
    ga = jnp.squeeze(dataset.posterior["ga"].values, axis=0)
    ga = jnp.concatenate((ga, jnp.ones(ga.shape[0])[:,None]), axis=1)
    
    seq_names = LD.seq_names
    N_variant = len(seq_names)
    
    # Loop over ga and make relative rel_to
    for i,s in enumerate(seq_names):
        if s == rel_to:
            ga = jnp.divide(ga, ga[:,i][:, None])

    # Compute medians and quantiles
    meds = jnp.median(ga, axis=0)
    gas = []
    for i,p in enumerate(ps):
        up = 0.5 + p/2
        lp = 0.5 - p/2
        gas.append(jnp.quantile(ga, jnp.array([lp, up]), axis=0).T)
    
    # Make empty dictionary
    v_dict = dict()
    v_dict["location"] = []
    v_dict["variant"] = []
    v_dict["median_ga"] = []
    
    for p in ps:
        v_dict[f"ga_upper_{round(p * 100)}"] = []
        v_dict[f"ga_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        if seq_names[variant] != rel_to:
            v_dict["location"].append(name)
            v_dict["variant"].append(seq_names[variant])
            v_dict["median_ga"].append(meds[variant])
            for i,p in enumerate(ps):
                v_dict[f"ga_upper_{round(p * 100)}"].append(gas[i][variant, 1])
                v_dict[f"ga_lower_{round(p * 100)}"].append(gas[i][variant, 0])

    return v_dict

def get_growth_advantage_time(dataset, LD, ps, name, rel_to="other"):
    ga = jnp.squeeze(dataset.posterior["ga"].values, axis=0)
    freq_medians = dataset.posterior["freq"].median(dim="draw").values[0]
    ga = jnp.concatenate((ga, jnp.ones(ga.shape[:2])[:, :, None]), axis=-1)

    seq_names = LD.seq_names
    N_variant = len(seq_names)
    T = len(LD.dates)
    
    # Loop over ga and make relative rel_to
    for i,s in enumerate(seq_names):
        if s == rel_to:
            ga = jnp.divide(ga, ga[:,:, i][:, :, None])

    # Compute medians and quantiles
    ga_meds = jnp.median(ga,axis=0)
    gas = []
    for i,p in enumerate(ps):
        up = 0.5 + p/2
        lp = 0.5 - p/2
        gas.append(jnp.quantile(ga, jnp.array([lp, up]), axis=0).T)
    
    # Make empty dictionary
    v_dict = dict()
    v_dict["date"] = []
    v_dict["location"] = []
    v_dict["variant"] = []
    v_dict["median_ga"] = []
    v_dict["median_freq"] = []
   
    for p in ps:
        v_dict[f"ga_upper_{round(p * 100)}"] = []
        v_dict[f"ga_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        if seq_names[variant] != rel_to:
            v_dict["date"] += LD.dates
            v_dict["location"] += [name]*T
            v_dict["variant"] += [seq_names[variant]] * T
            v_dict["median_ga"] += list(ga_meds[:,variant])
            v_dict["median_freq"] += list(freq_medians[:,variant])
            for i,p in enumerate(ps):
                v_dict[f"ga_upper_{round(p * 100)}"] += list(gas[i][variant, :, 1])
                v_dict[f"ga_lower_{round(p * 100)}"] += list(gas[i][variant, :, 0])
    return v_dict

def get_R(dataset, LD, ps, name, forecast=False):
    var_name = "R"
    f_name = "freq"
    if forecast:
        var_name += "_forecast"
        f_name += "_forecast"

    R_medians = dataset.posterior[var_name].median(dim="draw").values[0]
    freq_medians = dataset.posterior[f_name].median(dim="draw").values[0]
    N_variant = R_medians.shape[1]
    T = R_medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates
    if forecast:
        dates = forecast_dates(dates, T)

    R = []
    for i in range(len(ps)):
        R.append(jnp.array(az.hdi(dataset, var_names=var_name, hdi_prob=ps[i])[var_name]))

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

def get_little_r(dataset, g, LD, ps, name, forecast=False):
    var_name = "R"
    f_name = "freq"
    if forecast:
        var_name += "_forecast"
        f_name += "_forecast"

    # Get generation time
    mn = np.sum([p * (x+1) for x, p in enumerate(g)]) # Get mean of discretized generation time
    sd = np.sqrt(np.sum([p * (x+1) **2 for x, p in enumerate(g)])-mn**2) # Get sd of discretized generation time
    e_ = sd**2 / mn**2
    l = mn / (sd**2)
    
    # Set up conversion
    def _to_little_r(R):
        return (jnp.float_power(R, e_) - 1) * l
    
    R_medians = dataset.posterior[var_name].median(dim="draw").values[0]
    freq_medians = dataset.posterior[f_name].median(dim="draw").values[0]
    N_variant = R_medians.shape[1]
    T = R_medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates
    if forecast:
        dates = forecast_dates(LD.dates, T)
    
    R = []
    for i in range(len(ps)):
        R.append(jnp.array(az.hdi(dataset, var_names=var_name, hdi_prob=ps[i])[var_name]))

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

def get_I(dataset, LD, ps, name, forecast=False):
    var_name = "I_smooth"
    if forecast:
        var_name = "I_forecast"

    medians = jnp.round(dataset.posterior[var_name].median(dim="draw").values[0])
    N_variant = medians.shape[1]
    T = medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates
    if forecast:
        dates = forecast_dates(LD.dates, T)

    I = []
    for i in range(len(ps)):
        I.append(jnp.rint(jnp.array(az.hdi(dataset, var_names=var_name, hdi_prob=ps[i])[var_name])))

    I_dict = dict()
    I_dict["date"] = []
    I_dict["location"] = []
    I_dict["variant"] = []
    I_dict["median_I"] = []
        
    for p in ps:
        I_dict[f"I_upper_{round(p * 100)}"] = []
        I_dict[f"I_lower_{round(p * 100)}"] = []
        
    for variant in range(N_variant):
        I_dict["date"] += list(dates)
        I_dict["location"] += [name] * T
        I_dict["variant"] += [seq_names[variant]] * T
        I_dict["median_I"] += list(medians[:, variant])
        for i,p in enumerate(ps):
            I_dict[f"I_upper_{round(ps[i] * 100)}"] += list(I[i][:, variant, 1])
            I_dict[f"I_lower_{round(ps[i] * 100)}"] += list(I[i][:, variant, 0])

    return I_dict

def get_freq(dataset, LD, ps, name, forecast=False):
    var_name = "freq"
    if forecast:
        var_name = "freq_forecast"

    medians = dataset.posterior[var_name].median(dim="draw").values[0]
    N_variant = medians.shape[1]
    T = medians.shape[0]

    seq_names = LD.seq_names
    dates = LD.dates
    if forecast:
        dates = forecast_dates(LD.dates, T)

    P = []
    for i in range(len(ps)):
        P.append(jnp.array(az.hdi(dataset, var_names=var_name, hdi_prob=ps[i])[var_name]))

    freq_dict = dict()
    freq_dict["date"] = []
    freq_dict["location"] = []
    freq_dict["variant"] = []
    freq_dict["median_freq"] = []
        
    for p in ps:
        freq_dict[f"freq_upper_{round(p * 100)}"] = []
        freq_dict[f"freq_lower_{round(p * 100)}"] = []
        
    for variant in range(N_variant):
        freq_dict["date"] += list(dates)
        freq_dict["location"] += [name] * T
        freq_dict["variant"] += [seq_names[variant]] * T
        freq_dict["median_freq"] += list(medians[:, variant])
        for i,p in enumerate(ps):
            freq_dict[f"freq_upper_{round(ps[i] * 100)}"] += list(P[i][:, variant, 1])
            freq_dict[f"freq_lower_{round(ps[i] * 100)}"] += list(P[i][:, variant, 0])

    return freq_dict
