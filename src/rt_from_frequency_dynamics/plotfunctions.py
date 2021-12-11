import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Plot aes
class DefaultAes():
    ps = [0.95, 0.8, 0.5]
    alphas = [0.2, 0.4, 0.6]
    lineage_colors = ["#2e5eaa", "#5adbff",  "#56e39f","#b4c5e4", "#f03a47",  "#f5bb00", "#9e4244", "#808080"]


def define_color_map(color, seq_names):
    return {s:c for c,s in zip(color, seq_names)}

def get_quantile(dataset, p, var):
    return jnp.array(az.hdi(dataset, var_names=var, hdi_prob=p)[var])

def get_median(dataset, var):
    return dataset.posterior[var].median(dim="draw").values[0]

def get_quants(dataset, ps, var):
    V = []
    for i in range(len(ps)):
        V.append(get_quantile(dataset, ps[i], var))    
    med = get_median(dataset, var)
    return med, V

def plot_R(ax, dataset, ps, alphas, colors):
    med, R =  get_quants(dataset, ps, "R")
    t = jnp.arange(0, R[-1].shape[0], 1)
    N_lineage = R[-1].shape[1]
    
    # Make figure
    ax.axhline(y=1.0, color='k', linestyle='--')
    for lineage in range(N_lineage):
        for i in range(len(ps)):
            ax.fill_between(t, R[i][:, lineage, 0], R[i][:, lineage, 1],
                            color=colors[lineage], alpha=alphas[i])
        ax.plot(t, med[:, lineage],
                color=colors[lineage])

def plot_R_censored(ax, dataset, ps, alphas, colors, thres=0.001):
    med, R =  get_quants(dataset, ps, "R")
    t = jnp.arange(0, R[-1].shape[0], 1)
    N_lineage = R[-1].shape[1]
    med_freq = get_median(dataset, "freq")
    
    # Make figure
    ax.axhline(y=1.0, color='k', linestyle='--')
    for lineage in range(N_lineage):
        include = med_freq[:, lineage] > thres
        for i in range(len(ps)):
            ax.fill_between(t[include], R[i][include, lineage, 0], R[i][include, lineage, 1],
                            color=colors[lineage], alpha=alphas[i])
        ax.plot(t[include], med[include, lineage],
                color=colors[lineage])                

def plot_posterior_average_R(ax, dataset, ps, alphas, color):
    med, V =  get_quants(dataset, ps, "R_ave")
    t = jnp.arange(0, V[-1].shape[0], 1)
    
    # Make figure
    ax.axhline(y=1.0, color='k', linestyle='--')
    for i in range(len(ps)):
        ax.fill_between(t, V[i][:, 0], V[i][:, 1], color=color, alpha = alphas[i])
    ax.plot(t, med, color=color)

def plot_little_r_censored(ax, dataset, g, ps, alphas, colors, thres=0.001):
    med, R =  get_quants(dataset, ps, "R")
    t = jnp.arange(0, R[-1].shape[0], 1)
    N_lineage = R[-1].shape[1]
    med_freq = get_median(dataset, "freq")
    
    # Get generation time
    mn = np.sum([p * (x+1) for x, p in enumerate(g)]) # Get mean of discretized generation time
    sd = np.sqrt(np.sum([p * (x+1) **2 for x, p in enumerate(g)])-mn**2) # Get sd of discretized generation time
    e_ = sd**2 / mn**2
    l = mn / (sd**2)
    
    def _to_little_r(R):
        return (jnp.float_power(R, e_) - 1) * l
        
    # Make figure
    ax.axhline(y=0.0, color='k', linestyle='--')
    for lineage in range(N_lineage):
        include = med_freq[:, lineage] > thres
        for i in range(len(ps)):
            ax.fill_between(t[include], 
                            _to_little_r(R[i][include, lineage, 0]), 
                            _to_little_r(R[i][include, lineage, 1]),
                            color=colors[lineage], alpha=alphas[i])
        ax.plot(t[include], _to_little_r(med[include, lineage]),
                color=colors[lineage])   

def plot_posterior_frequency(ax, dataset, ps, alphas, colors):
    med, V =  get_quants(dataset, ps, "freq")
    t = jnp.arange(0, V[-1].shape[0], 1)
    N_lineage = V[-1].shape[1]
    
    # Make figure
    for lineage in range(N_lineage):
        for i in range(len(ps)):
            ax.fill_between(t, V[i][:, lineage, 0], V[i][:, lineage, 1],
                            color=colors[lineage], alpha=alphas[i])
        ax.plot(t, med[:, lineage],
                color=colors[lineage])

def plot_observed_frequency(ax, LD, colors):
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_lineage = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for lineage in range(N_lineage):
        ax.scatter(t, obs_freq[:, lineage], color=colors[lineage],  edgecolor="black")

def plot_observed_frequency_size(ax, LD, colors, size):
    N = LD.seq_counts.sum(axis=1)[:, None] 
    sizes = [size(n) for n in N]
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    N_lineage = obs_freq.shape[-1]
    t = jnp.arange(0, obs_freq.shape[0])
    for lineage in range(N_lineage):
        ax.scatter(t, obs_freq[:, lineage], color=colors[lineage], s=sizes, edgecolor="black")

def plot_posterior_I(ax, dataset, ps, alphas, colors):
    med, V =  get_quants(dataset, ps, "I_smooth")
    t = jnp.arange(0, V[-1].shape[0], 1)
    N_lineage = V[-1].shape[1]
    
    # Make figure
    for lineage in range(N_lineage):
        for i in range(len(ps)):
            ax.fill_between(t, V[i][:, lineage, 0], V[i][:, lineage, 1],
                            color=colors[lineage], alpha=alphas[i])
        ax.plot(t, med[:, lineage],
                color=colors[lineage])

def plot_posterior_smooth_EC(ax, dataset, ps, alphas, color):
    med, V =  get_quants(dataset, ps, "total_smooth_prev")
    t = jnp.arange(0, V[-1].shape[0], 1)
    
    # Make figure
    for i in range(len(ps)):
        ax.fill_between(t, V[i][:, 0], V[i][:, 1], color=color, alpha = alphas[i])
    ax.plot(t, med, color=color)

def plot_cases(ax, LD):
    t = jnp.arange(0, LD.cases.shape[0])
    ax.bar(t, LD.cases, color = "black", alpha = 0.3)

def add_dates(ax, dates, sep=1):
    t = []
    labels = []
    for (i, date) in enumerate(dates):
        if int(date.strftime("%d")) == 1:
            labels.append(date.strftime("%b"))
            t.append(i)
    ax.set_xticks(t[::sep])
    ax.set_xticklabels(labels[::sep])

def plot_growth_advantage(ax, dataset, LD, ps, alphas, colors):
    ga = jnp.array(dataset.posterior["ga"])[1]
    
    inds = jnp.arange(0, ga.shape[-1], 1)
    
    ax.axhline(y=1.0, color='k', linestyle='--')
    parts = ax.violinplot(ga.T, inds,
                         showmeans=False, showmedians=False, showextrema=False)
    
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        
    q1, med, q3 = jnp.percentile(ga, jnp.array([25, 50, 75]), axis=0)
    ax.scatter(inds, med, color="white", zorder=3, edgecolor="black")
    ax.vlines(inds, q1, q3, color = "k", lw=4, zorder=2)
    
    q1, med, q3 = jnp.percentile(ga, jnp.array([2.5, 50, 97.5]), axis=0)
    ax.vlines(inds, q1, q3, color = "k", lw=2, zorder=1)
    
    ax.set_xticks(inds)
    ax.set_xticklabels(LD.seq_names[:-1])

def plot_total_by_obs_frequency(ax, LD, total, colors):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    obs_freq = jnp.divide(LD.seq_counts, LD.seq_counts.sum(axis=1)[:, None])
    
    # Make figure
    bottom = jnp.zeros(t.shape)
    for lineage in range(D):
        ax.bar(t, obs_freq[:, lineage] * total, bottom = bottom,
                color=colors[lineage])
        bottom = obs_freq[:, lineage] * total + bottom
        
def plot_total_by_median_frequency(ax, dataset, LD, total, colors):
    T, D = LD.seq_counts.shape
    t = jnp.arange(0, T, 1)
    med_freq = get_median(dataset, "freq")
    
    # Make figure
    bottom = jnp.zeros(t.shape)
    for lineage in range(D):
        ax.bar(t, med_freq[:, lineage] * total, bottom = bottom,
                color=colors[lineage])
        bottom = med_freq[:, lineage] * total + bottom
