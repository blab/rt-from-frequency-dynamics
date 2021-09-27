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
