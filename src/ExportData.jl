function get_Rt_dataframe(MS::ModelStan, LD::LineageData)
    ps = [0.5, 0.8, 0.95]
    seed_L = MS.data["seed_L"]
    N_lineage = MS.data["N_lineage"]
    model_name = MS.model.name
    seq_labels = LD.seq_names
    dates = LD.dates
    
    # Process Rt 
    R = get_posterior(MS, "R.", true)
    med, lQ, uQ = get_quants(R, ps)

    # Process frequencies
    sim_freq = get_posterior(MS, "sim_freq", true)
    freq_med, _, _ = get_quants(sim_freq, ps)

    # Allocate columns
    dates_vec = [] 
    state_vec = []
    lineage_vec = []
    rt_median = []
    rt_lower = [[] for i in 1:length(ps)]
    rt_upper = [[] for i in 1:length(ps)]
    freq_median = []

    for lineage in 1:N_lineage
        dates_vec = vcat(dates_vec, dates)
        state_vec = vcat(state_vec, repeat([model_name], length(dates)))
        lineage_vec = vcat(lineage_vec, repeat([seq_labels[lineage]], length(dates)))
        rt_median = vcat(rt_median, med[:, lineage])
        freq_median = vcat(freq_median, freq_med[:, lineage])
        for i in 1:length(ps)
            rt_lower[i] = vcat(rt_lower[i], lQ[i][:, lineage]) # Each col different p
            rt_upper[i] = vcat(rt_upper[i], uQ[i][:, lineage])
        end
    end
    
    rt_lower = vcat(rt_lower...)
    rt_upper = vcat(rt_upper...)

    return DataFrame(state = state_vec, 
        date = dates_vec,
        lineage = lineage_vec,
        rt_median = rt_median,
        rt_lower_50 = rt_lower[1],
        rt_upper_50 = rt_upper[1],
        rt_lower_80 = rt_lower[2],
        rt_upper_80 = rt_upper[2],
        rt_lower_95 = rt_lower[3],
        rt_upper_95 = rt_upper[3],
        freq_median = freq_median)
end


function get_growth_advantages_dataframe(MS::ModelStan, LD::LineageData)
    seq_labels = LD.seq_names

    v = get_posterior(MS, "v", false)
    v = [exp.(vi) for vi in v]
    med, lQ, uQ = get_quants(v, ps)


    state_vec = repeat([MS.model.name], length(med))
    lineage_vec =  vcat(seq_labels)[1:end-1]

    return DataFrame(state = state_vec, 
               lineage = lineage_vec,
               v_median = vec(med),
               v_lower_50 = lQ[1],
               v_upper_50 = uQ[1],
               v_lower_80 = lQ[2],
               v_upper_80 = uQ[2],
               v_lower_95 = lQ[3],
               v_upper_95 = uQ[3]
               )    
end