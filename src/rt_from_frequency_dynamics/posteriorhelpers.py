import jax.numpy as jnp

from .datahelpers import forecast_dates


def get_site_by_variant(dataset, LD, ps, name, site, forecast=False):

    # Unpack variant info
    seq_names = LD.seq_names
    dates = LD.dates

    # Unpack posterior
    var_name = site + "_forecast" if forecast else site
    var = dataset[var_name]
    N_variant = var.shape[-1]
    T = var.shape[-2]

    if forecast:
        dates = forecast_dates(dates, T)

    # Compute medians and hdis for ps
    var_median = jnp.median(var, axis=0)

    var_hdis = [
        jnp.quantile(var, q=jnp.array([0.5 * (1 - p), 0.5 * (1 + p)]), axis=0)
        for i, p in enumerate(ps)
    ]

    var_dict = dict()
    var_dict["date"] = []
    var_dict["location"] = []
    var_dict["variant"] = []
    var_dict[f"median_{var_name}"] = []
    for p in ps:
        var_dict[f"{var_name}_upper_{round(p * 100)}"] = []
        var_dict[f"{var_name}_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        var_dict["date"] += list(dates)
        var_dict["location"] += [name] * T
        var_dict["variant"] += [seq_names[variant]] * T
        var_dict[f"median_{var_name}"] += list(var_median[:, variant])
        for i, p in enumerate(ps):
            var_dict[f"{var_name}_upper_{round(ps[i] * 100)}"] += list(
                var_hdis[i][1, :, variant]
            )
            var_dict[f"{var_name}_lower_{round(ps[i] * 100)}"] += list(
                var_hdis[i][0, :, variant]
            )
    return var_dict


def add_median_freq(var_dict, dataset, forecast=False, excluded=None):
    f_name = "freq"
    f_name = f_name + "_forecast" if forecast else f_name

    freq = dataset[f_name]
    freq_median = jnp.median(freq, axis=0)
    var_dict["median_freq"] = []
    for variant in range(freq_median.shape[-1]):
        if variant != excluded:
            var_dict["median_freq"] += list(freq_median[:, variant])
    return var_dict


def get_growth_advantage(dataset, LD, ps, name, rel_to="other"):
    # Unpack variant info
    seq_names = LD.seq_names

    # Get posterior samples
    ga = dataset["ga"]
    ga = jnp.concatenate((ga, jnp.ones(ga.shape[0])[:, None]), axis=1)
    N_variant = ga.shape[-1]

    # Loop over ga and make relative rel_to
    for i, s in enumerate(seq_names):
        if s == rel_to:
            ga = jnp.divide(ga, ga[:, i][:, None])

    # Compute medians and quantiles
    meds = jnp.median(ga, axis=0)
    gas = []
    for i, p in enumerate(ps):
        up = 0.5 + p / 2
        lp = 0.5 - p / 2
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
            for i, p in enumerate(ps):
                v_dict[f"ga_upper_{round(p * 100)}"].append(gas[i][variant, 1])
                v_dict[f"ga_lower_{round(p * 100)}"].append(gas[i][variant, 0])

    return v_dict


def get_growth_advantage_time(dataset, LD, ps, name, rel_to="other"):
    # Unpack data
    seq_names = LD.seq_names
    N_variant = len(seq_names)
    T = len(LD.dates)

    # Unpack posterior
    ga = dataset["ga"]
    ga = jnp.concatenate((ga, jnp.ones(ga.shape[:2])[:, :, None]), axis=-1)
    # Loop over ga and make relative rel_to
    for i, s in enumerate(seq_names):
        if s == rel_to:
            ga = jnp.divide(ga, ga[..., i][..., None])

    # Compute medians and quantiles
    ga_meds = jnp.median(ga, axis=0)
    gas = [
        jnp.quantile(ga, q=jnp.array([0.5 * (1 - p), 0.5 * (1 + p)]), axis=0)
        for i, p in enumerate(ps)
    ]

    # Make empty dictionary
    v_dict = dict()
    v_dict["date"] = []
    v_dict["location"] = []
    v_dict["variant"] = []
    v_dict["median_ga"] = []

    for p in ps:
        v_dict[f"ga_upper_{round(p * 100)}"] = []
        v_dict[f"ga_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        if seq_names[variant] != rel_to:
            v_dict["date"] += LD.dates
            v_dict["location"] += [name] * T
            v_dict["variant"] += [seq_names[variant]] * T
            v_dict["median_ga"] += list(ga_meds[:, variant])
            for i, p in enumerate(ps):
                v_dict[f"ga_upper_{round(p * 100)}"] += list(gas[i][1, :, variant])
                v_dict[f"ga_lower_{round(p * 100)}"] += list(gas[i][0, :, variant])
        else:
            excluded = variant
    v_dict = add_median_freq(v_dict, dataset, excluded=excluded)
    return v_dict


def get_R(dataset, LD, ps, name, forecast=False):
    var_dict = get_site_by_variant(dataset, LD, ps, name, "R", forecast=forecast)
    return add_median_freq(var_dict, dataset, forecast)


def get_little_r(dataset, LD, ps, name, forecast=False):
    var_dict = get_site_by_variant(dataset, LD, ps, name, "r", forecast=forecast)
    return add_median_freq(var_dict, dataset, forecast)


def get_I(dataset, LD, ps, name, forecast=False):
    var_dict = get_site_by_variant(dataset, LD, ps, name, "I_smooth", forecast=forecast)
    return add_median_freq(var_dict, dataset, forecast)


def get_freq(dataset, LD, ps, name, forecast=False):
    return get_site_by_variant(dataset, LD, ps, name, "freq", forecast=forecast)
