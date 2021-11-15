# Interval Based
"""
Takes in array of matrices each of which is a simulation. Returns lQ, uQ, med which are the lQuant, uQuant quantiles and median respectively.
"""
function sim_stats(sims, lQuant = 0.05, uQuant = 0.95)
    n_sims = length(sims)
    t_max, n_var = size(sims[1])
    uQ = zeros(t_max, n_var)
    lQ = zeros(t_max, n_var)
    med = zeros(t_max, n_var)
    
    for t in 1:t_max
        for k in 1:n_var
            time_slice = [sims[i][t,k] for i in 1:n_sims]
            uQ[t,k] = quantile(time_slice, uQuant)
            lQ[t,k] = quantile(time_slice, lQuant)
            med[t,k] = median(time_slice)
        end
    end
    return lQ, uQ, med
end


#TODO: Just use multiple dispatch for sim_stats
"""
Runs sim_stats for multiple quantiles.
"""
function sim_stats_multi(Q, lQuant, uQuant)
    @assert length(lQuant) == length(uQuant)

    tmp = [sim_stats(Q, lQuant[i], uQuant[i]) for i in 1:length(lQuant)]

    lQ = [tmp[i][1] for i in 1:length(lQuant)]
    uQ = [tmp[i][2] for i in 1:length(lQuant)]
    med = [tmp[i][3] for i in 1:length(lQuant)]

    return lQ, uQ, med
end

"""
Sample the posterior of `varname` in `sim` with label names `cnames`
"""
function sample_posterior(sim, cnames, varname)

    # Combine various chains 
    _, n_vars, n_chains = size(sim)
    sim = vcat([sim[:, :, n] for n in 1:n_chains]...)
    N_samples, n_vars = size(sim)
    
    # Get indices for the variables of interest
    #id = occursin.(varname,cnames)
    id = startswith.(cnames, varname)

    # Get Vector of Matrices for Quantity of Interest
    Q = []

    for sample in 1:N_samples
        push!(Q, sim[sample, id])
    end

    return Q
end

function sample_posterior(sim, cnames, N_demes, varname)
    
    # Combine various chains 
    _, n_vars, n_chains = size(sim)
    sim = vcat([sim[:, :, n] for n in 1:n_chains]...)
    N_samples, n_vars = size(sim)
    
    # Get indices for the variables of interest
    id = []
    for deme in 1:N_demes
        #push!(id, occursin.(varname,cnames) .& endswith.(cnames, ".$deme"))
        push!(id, startswith.(cnames, varname) .& endswith.(cnames, ".$deme"))
    end
    id = hcat(id...)
    
    # Get Vector of Matrices for Quantity of Interest
    Q = Matrix{Float64}[]
    
    for sample in 1:N_samples
        this_sample = []
        for deme in 1:N_demes
           push!(this_sample, sim[sample, id[:,deme]])
        end
        push!(Q, hcat(this_sample...))
    end
    
    return Q
end

function get_posterior(MS::ModelStan, var::String, multi::Bool)
    if multi
        sample_posterior(MS.posterior["samples"], MS.posterior["cnames"], MS.data["N_lineage"], var)
    else
        sample_posterior(MS.posterior["samples"], MS.posterior["cnames"], var)
    end
end

"""
Restructure quantiles so that lQ, uQ, med are indexed by deme.
"""
function parse_by_deme(lQ, uQ, med)
    len_quants = size(med)[1]
    L, N_deme = size(med[1])

    # Put here
    lQ = [hcat([lQ[ints][:, i] for ints in 1:len_quants]...) for i in 1:N_deme]
    uQ = [hcat([uQ[ints][:, i] for ints in 1:len_quants]...) for i in 1:N_deme]
    med = [hcat([med[ints][:, i] for ints in 1:len_quants]...) for i in 1:N_deme]

    return lQ, uQ, med
end

# HDI-Based
function reshape_to_original(Q)
    # Q is vector of samples
    shape = size(Q[1])
    sample_type = eltype(Q[1])
    
    # Q_ has original shape of quantity 
    # Each index is a vector of samples
    Q_ = Array{Vector{sample_type}}(undef, size(Q[1])...)
    N = length(Q)
    
    for idx in eachindex(Q_)
       Q_[idx] = sample_type[Q[sample][idx] for sample in 1:N]
    end
    
    return Q_
end

# From MCMCChains
function hpd(x::AbstractVector{<:Real}; alpha::Real=0.05)
    n = length(x)
    m = max(1, ceil(Int, alpha * n))

    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end

lower_hpd(x;alpha=0.05) = hpd(x;alpha=alpha)[1]
upper_hpd(x;alpha=0.05) = hpd(x;alpha=alpha)[2]

"""
Given quantitiy Q (Vector of samples of Q)
return median, lower and upper HPD bounds for probability ps.
Each lQ, uQ has indices so that ps[i] -> lQ[i].
lQ[i][idx] gives lower bound of ps[i] of Q[i,j] from Stan.
"""
function get_quants(Q, ps)
    Q_ = reshape_to_original(Q)
    med = median.(Q_)
    lQ = []
    uQ = []
    for p in ps
        push!(lQ, lower_hpd.(Q_, alpha = 1-p)) 
        push!(uQ, upper_hpd.(Q_, alpha = 1-p)) 
    end
    
    return med, lQ, uQ
end
