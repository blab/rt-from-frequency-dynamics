import jax.numpy as jnp
import datetime


class DefaultAes:
    ps = [0.95, 0.8, 0.5]
    alphas = [0.2, 0.4, 0.6]
    variant_colors = [
        "#2e5eaa",
        "#5adbff",
        "#56e39f",
        "#b4c5e4",
        "#f03a47",
        "#f5bb00",
        "#9e4244",
        "#808080",
    ]


def define_color_map(color, seq_names):
    return {s: c for c, s in zip(color, seq_names)}


def expand_dates(dates, T_forecast):
    x_dates = dates.copy()
    for d in range(T_forecast):
        x_dates.append(dates[-1] + datetime.timedelta(days=d + 1))
    return x_dates


def get_quantile(dataset, p, var):
    q = jnp.array([0.5 * (1 - p), 0.5 * (1 + p)])
    return jnp.quantile(dataset[var], q=q, axis=0)


def get_median(dataset, var):
    return jnp.median(dataset[var], axis=0)


def get_quantiles(dataset, ps, var):
    V = []
    for i in range(len(ps)):
        V.append(get_quantile(dataset, ps[i], var))
    med = get_median(dataset, var)
    return med, V


def prep_posterior_for_plot(var, dataset, ps, forecast=False):
    """
    Prepare posteriors for plotting by finding time span, medians, and quantiles.
    """
    med, quants = get_quantiles(dataset, ps, var)
    t = jnp.arange(0, med.shape[0], 1)

    if forecast:
        med_f, quants_f = get_quantiles(dataset, ps, var + "_forecast")
        t_f = med.shape[0] + jnp.arange(0, med_f.shape[0], 1)
        t = jnp.concatenate((t, t_f))
        med = jnp.concatenate((med, med_f))
        for i in range(len(ps)):
            quants[i] = jnp.concatenate([quants[i], quants_f[i]], axis=1)
    return t, med, quants


def plot_posterior_time(ax, t, med, quants, alphas, colors, included=None):
    """
    Loop over variants to plot medians and quantiles at specifed points.
    Plots all time points unless time points to be included are specified in included.
    """

    for variant in range(med.shape[-1]):
        v_included = (
            jnp.arange(0, med.shape[0]) if included is None else included[:, variant]
        )
        for i in range(len(quants)):
            ax.fill_between(
                t[v_included],
                quants[i][0, v_included, variant],
                quants[i][1, v_included, variant],
                color=colors[variant],
                alpha=alphas[i],
            )
        ax.plot(t[v_included], med[v_included, variant], color=colors[variant])


def plot_R(ax, dataset, ps, alphas, colors, forecast=False):
    t, med, quants = prep_posterior_for_plot("R", dataset, ps, forecast=forecast)
    ax.axhline(y=1.0, color="k", linestyle="--")
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_R_censored(ax, dataset, ps, alphas, colors, forecast=False, thres=0.001):
    t, med, quants = prep_posterior_for_plot("R", dataset, ps, forecast=forecast)
    ax.axhline(y=1.0, color="k", linestyle="--")

    # Plot only variants at high enough frequency
    _, freq_median, _ = prep_posterior_for_plot("freq", dataset, ps, forecast=forecast)
    included = freq_median > thres

    plot_posterior_time(ax, t, med, quants, alphas, colors, included=included)


def plot_posterior_average_R(ax, dataset, ps, alphas, color):
    med, V = get_quantiles(dataset, ps, "R_ave")
    t = jnp.arange(0, V[-1].shape[0], 1)

    # Make figure
    ax.axhline(y=1.0, color="k", linestyle="--")
    for i in range(len(ps)):
        ax.fill_between(t, V[i][:, 0], V[i][:, 1], color=color, alpha=alphas[i])
    ax.plot(t, med, color=color)


def plot_little_r_censored(
    ax, dataset, ps, alphas, colors, forecast=False, thres=0.001
):
    t, med, quants = prep_posterior_for_plot("r", dataset, ps, forecast=forecast)
    ax.axhline(y=0.0, color="k", linestyle="--")

    # Plot only variants at high enough frequency
    _, freq_median, _ = prep_posterior_for_plot("freq", dataset, ps, forecast=forecast)
    included = freq_median > thres

    plot_posterior_time(ax, t, med, quants, alphas, colors, included=included)


def plot_posterior_frequency(ax, dataset, ps, alphas, colors, forecast=False):
    t, med, quants = prep_posterior_for_plot("freq", dataset, ps, forecast=forecast)
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_observed_frequency(ax, LD, colors):
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_variant = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for variant in range(N_variant):
        ax.scatter(t, obs_freq[:, variant], color=colors[variant], edgecolor="black")


def plot_observed_frequency_size(ax, LD, colors, size):
    N = LD.seq_counts.sum(axis=1)[:, None]
    sizes = [size(n) for n in N]
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_variant = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for variant in range(N_variant):
        ax.scatter(
            t, obs_freq[:, variant], color=colors[variant], s=sizes, edgecolor="black"
        )


def plot_posterior_I(ax, dataset, ps, alphas, colors, forecast=False):
    t, med, quants = prep_posterior_for_plot("I_smooth", dataset, ps, forecast=forecast)
    plot_posterior_time(ax, t, med, quants, alphas, colors)


def plot_posterior_smooth_EC(ax, dataset, ps, alphas, color):
    med, V = get_quantiles(dataset, ps, "total_smooth_prev")
    t = jnp.arange(0, V[-1].shape[0], 1)

    # Make figure
    for i in range(len(ps)):
        ax.fill_between(t, V[i][:, 0], V[i][:, 1], color=color, alpha=alphas[i])
    ax.plot(t, med, color=color)


def plot_cases(ax, LD):
    t = jnp.arange(0, LD.cases.shape[0])
    ax.bar(t, LD.cases, color="black", alpha=0.3)


def add_dates(ax, dates, sep=1):
    t = []
    labels = []
    for (i, date) in enumerate(dates):
        if int(date.strftime("%d")) == 1:
            labels.append(date.strftime("%b"))
            t.append(i)
    ax.set_xticks(t[::sep])
    ax.set_xticklabels(labels[::sep])


def add_dates_sep(ax, dates, sep=7):
    t = []
    labels = []
    for (i, date) in enumerate(dates):
        if (i % sep) == 0:
            labels.append(date.strftime("%b %d"))
            t.append(i)
    ax.set_xticks(t)
    ax.set_xticklabels(labels)


def plot_growth_advantage(ax, dataset, LD, ps, alphas, colors):
    ga = jnp.array(dataset.posterior["ga"])[1]

    inds = jnp.arange(0, ga.shape[-1], 1)

    ax.axhline(y=1.0, color="k", linestyle="--")
    parts = ax.violinplot(
        ga.T, inds, showmeans=False, showmedians=False, showextrema=False
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    q1, med, q3 = jnp.percentile(ga, jnp.array([25, 50, 75]), axis=0)
    ax.scatter(inds, med, color="white", zorder=3, edgecolor="black")
    ax.vlines(inds, q1, q3, color="k", lw=4, zorder=2)

    q1, med, q3 = jnp.percentile(ga, jnp.array([2.5, 50, 97.5]), axis=0)
    ax.vlines(inds, q1, q3, color="k", lw=2, zorder=1)

    ax.set_xticks(inds)
    ax.set_xticklabels(LD.seq_names[:-1])


def plot_total_by_obs_frequency(ax, LD, total, colors):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    obs_freq = jnp.nan_to_num(
        jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    )

    # Make figure
    bottom = jnp.zeros(t.shape)
    for variant in range(D):
        ax.bar(t, obs_freq[:, variant] * total, bottom=bottom, color=colors[variant])
        bottom = obs_freq[:, variant] * total + bottom


def plot_total_by_median_frequency(ax, dataset, LD, total, colors):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    med_freq = get_median(dataset, "freq")

    # Make figure
    bottom = jnp.zeros(t.shape)
    for variant in range(D):
        ax.bar(t, med_freq[:, variant] * total, bottom=bottom, color=colors[variant])
        bottom = med_freq[:, variant] * total + bottom
