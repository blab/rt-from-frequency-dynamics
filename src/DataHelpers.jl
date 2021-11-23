function export_dataframe_for_modeling(cases, counts, lineage_names, state_name; start_date = Date(2020,1,1)) 
    dates = Day.(collect(0:(T-1))) .+ start_date
    case_df = DataFrame(state = state_name, cases = cases, date = dates)
    aug_lineage_names = vcat("total", lineage_names)
    aug_counts = Int.(hcat(sum(counts, dims=2), counts))
    count_df = DataFrame(aug_counts, "sequences_" .* aug_lineage_names)
    
    return hcat(case_df, count_df) 
end

# NEW LOADERS
function format_sequence_names(seq_names)
    # Move 'other' column to end
    if "other" in seq_names
        _tmp = String[]
        for s in seq_names
            if s != "other"
                push!(_tmp, s)
            end
        end
        push!(_tmp, "other")
        seq_names = _tmp
    end
    return seq_names
end

function counts_to_matrix(seq_df::DataFrame, seq_names)
    dmn, dmx = extrema(unique(seq_df.date)) # Get extrema
    dates = [dmn + Day(d) for d in 0:Dates.value(dmx-dmn)] # Generate intermediate dates   
    date_to_index = Dict(d => i for (i,d) in enumerate(dates)) # Turn dates to indices
    
    C = Vector{Int}[]
    for s in seq_names # for each variant
        C_s = zeros(Int, length(dates)) # Init counts at zero
        D_s = filter(row -> row.variant == s, seq_df) # Get variant df
        # Loop over dates and pull out corresponding count
        for row in eachrow(D_s)
            C_s[date_to_index[row.date]] += row.sequences
        end
        push!(C, C_s)
    end
    return dates, reduce(hcat, C) 
end

function prep_sequence_counts(seq_df::DataFrame)
    seq_names = format_sequence_names(unique(seq_df.variant)) # Get formated variant names
    dates, C = counts_to_matrix(seq_df, seq_names) # Count by lineage values per date
    return seq_names, dates, C
end

function prep_cases(cases_df::DataFrame)
    dmn, dmx = extrema(unique(cases_df.date)) # Get extrema
    dates = [dmn + Day(d) for d in 0:Dates.value(dmx-dmn)] # Generate intermediate dates   
    date_to_index = Dict(d => i for (i,d) in enumerate(dates)) # Turn dates to indices
    
    C = zeros(eltype(cases_df.cases), length(dates))
    for row in eachrow(cases_df)
        C[date_to_index[row.date]] += row.cases 
    end
    
    return cases_df.date, cases_df.cases
end

# ########

# function load_state_data(state, df)
#     df = sort!(df, :date) |>
#     filter(x -> x.state == state, df)
# end

function sequence_counts_to_matrix(state_df)
    seq_cols = sort([x for x in names(state_df) if startswith(x, "sequence") .& !endswith(x, "total") ]) #Check order
    seq_total = sort([x for x in names(state_df) if startswith(x, "sequence") .& endswith(x, "total") ]) #Check order
    #seq_cols = [x for x in names(state_df) if startswith(x, "sequence") ] #Check order
    return seq_cols, Matrix(state_df[:, seq_cols]), Vector(state_df[:, seq_total[1]])
end

function cases_to_vector(state_df)
    Vector(state_df[:, :cases])
end

# function get_data_obs_period(true_L, seed_L, forecast_L, is_sim)
#     # For simulations, we hold back more data for forecast evaluation and seeding
#     # For real data, we use all data for inference
#     if is_sim
#         L = true_L - forecast_L - seed_L
#         obs_period = (seed_L+1):(L + seed_L)
#     else
#         L = true_L
#         obs_period = 1:true_L
#     end
#     return L, obs_period
# end

function clean_labels(seq_labels, stan_data)
    reshape([split(label, "_")[2] for label in seq_labels],
                1, stan_data["N_lineage"])
end

# function get_data_for_inference(seed_L, forecast_L, 
#         C, sequence_counts, N_seqs, 
#         g, onset, model::TimeSeriesNode; is_sim=false)

#     true_L, N_deme = size(sequence_counts)

#     # How much data to pass to model
#     L, obs_period = get_data_obs_period(true_L, seed_L, forecast_L, is_sim)
    
#     t = collect(1:L)
#     X = get_design(model, t)
    
#     stan_data = Dict(
#         "seed_L" => seed_L,
#         "forecast_L" => forecast_L,
#         "L" => L,
#         "N_lineage" => N_deme,
#         "cases" => Int.(C)[obs_period],
#         "num_sequenced" => sequence_counts[obs_period, :],
#         "N_sequences" => Int.(N_seqs)[obs_period],
#         "g" => g,
#         "onset" => onset,
#         "l" => length(g),
#         "K" => size(X, 2),
#         "features" => X
#     )
#     return stan_data
# end

#  function process_state_data_for_stan(state, df, g, onset, seed_L, forecast_L, model::TimeSeriesNode)
#     state_df = load_state_data(state, df)
#     seq_cols, seq_counts, seq_total = sequence_counts_to_matrix(state_df)
#     cases = cases_to_vector(state_df)
    
#     stan_data = get_data_for_inference(seed_L, forecast_L,
#                                      cases, seq_counts, seq_total, 
#                                      g, onset, model)
    
#     return state_df[:, :date], seq_cols, stan_data
# end

# function define_stan_model(state, filename, 
#         model::TimeSeriesNode, priors, 
#         model_name; num_samples = 1000, num_warmup = 2000)
#     # Construct Model 
#     model_string = read(filename, String);
#     model_string = add_priors(model_string, model, priors)
#     model_string = add_other_parms(model_string, model, priors)
    
#     method = Sample(
#         save_warmup = false,
#         num_samples = 1000,
#         num_warmup = 2000)
    
#     #method = Variational(
#     #    grad_samples=1,
#      #   elbo_samples=100,
#       #  tol_rel_obj =1e-5,
#       #  output_samples=4000)
    
#     #method = Optimize(iter = 2000)
    
#     mkpath(cd(pwd, "..") * "/data/sims/$(model_name)/")
#     stan_model = Stanmodel(
#         method,
#         nchains = 4,
#         name = "rt-lineages-" * state,   
#         tmpdir = cd(pwd, "..") * "/data/sims/$(model_name)/" * state,
#         model = model_string); 
#  end

# function process_all_states(filename, df, g, onset, seed_L, forecast_L, 
#         model::TimeSeriesNode; 
#         priors = nothing,
#         state_names = unique(df[:, :state]),
#         model_name = "test",
#         num_samples = 1000,
#         num_warmup = 2000
#         )
#     states_dict = Dict()
    
#     for state in state_names
#         dates_vec, seq_labels, state_data = process_state_data_for_stan(state, df, g, onset, seed_L, forecast_L, model)
#         state = replace(state, ' ' => '_')
#         state_model = define_stan_model(state, filename, model, priors, model_name; 
#             num_samples = num_samples, num_warmup = num_warmup)
        
#        states_dict[state] = Dict(
#         "date" => dates_vec,
#         "seq_labels" => clean_labels(seq_labels, state_data),
#         "stan_data" => state_data,
#         "stan_model" => state_model
#         )
#     end
    
#     return states_dict    
# end

# Interval Based
# function get_Rt_by_state(state, states_dict)
#     N_lineages = states_dict[state]["stan_data"]["N_lineage"]
#     seq_labels = states_dict[state]["seq_labels"]
    
#     # Process Rt 
#     R = sample_posterior(states_dict[state]["stan_results"], states_dict[state]["stan_cnames"], N_lineages, "R.")
#     lQ, uQ, med = sim_stats_multi(R, lQuants, uQuants)
#     lQ, uQ, med = parse_by_deme(lQ, uQ, med)
    
#     sim_freq = sample_posterior(states_dict[state]["stan_results"], states_dict[state]["stan_cnames"], N_lineages, "sim_freq")
#     lQ_, uQ_, med_ = sim_stats_multi(sim_freq, lQuants, uQuants)
#     _, _, freq_med = parse_by_deme(lQ_, uQ_, med_)
    
#     dates_vec = [] 
#     state_vec = []
#     lineage_vec = []
#     rt_median = []
#     rt_lower = []
#     rt_upper = []
#     freq_median = []

#     # May have to adjust to the size of things
#     seed_L = states_dict[state]["stan_data"]["seed_L"]
#     for lineage in 1:N_lineages
#         dates_vec = vcat(dates_vec, states_dict[state]["date"])
#         state_vec = vcat(state_vec, repeat([state], length(states_dict[state]["date"])))
#         lineage_vec = vcat(lineage_vec, repeat([seq_labels[lineage]], length(states_dict[state]["date"])))
#         rt_median = vcat(rt_median, med[lineage][1:end, 1])
#         push!(rt_lower, lQ[lineage][1:end, :])
#         push!(rt_upper, uQ[lineage][1:end, :])
#         freq_median = vcat(freq_median, freq_med[lineage][1:end, 1])
#     end
    
#     rt_lower = vcat(rt_lower...)
#     rt_upper = vcat(rt_upper...)

#     return DataFrame(state = state_vec, 
#         date = dates_vec,
#         lineage = lineage_vec,
#         rt_median = rt_median,
#         rt_lower_50 = rt_lower[:, 1],
#         rt_upper_50 = rt_upper[:, 1],
#         rt_lower_80 = rt_lower[:, 2],
#         rt_upper_80 = rt_upper[:, 2],
#         rt_lower_95 = rt_lower[:, 3],
#         rt_upper_95 = rt_upper[:, 3],
#         freq_median = freq_median)
# end

# function get_growth_advantages(SoI, states_dict)
#     v = get_posterior(states_dict, SoI, "v", false)
#     v = exp.(vcat(hcat(v...)'))
#     med = median(v, dims = 1)
#     lQ = vcat([quantile(vi, lQuants) for vi in eachcol(v)]...)
#     uQ = vcat([quantile(vi, uQuants) for vi in eachcol(v)]...) 
    
#     state_vec = repeat([SoI], length(med))
#     lineage_vec =  vcat(states_dict[SoI]["seq_labels"]...)[1:end-1]

#     return DataFrame(state = state_vec, 
#                lineage = lineage_vec,
#                v_median = vec(med),
#                v_lower_50 = lQ[:, 1],
#                v_upper_50 = uQ[:, 1],
#                v_lower_80 = lQ[:, 2],
#                v_upper_80 = uQ[:, 2]
#                )    
# end


# # HDI-based
# function get_Rt_by_state_HDI(state, states_dict)
#     N_lineages = states_dict[state]["stan_data"]["N_lineage"]
#     seq_labels = states_dict[state]["seq_labels"]
    
#     # Process Rt 
#     R = sample_posterior(states_dict[state]["stan_results"], states_dict[state]["stan_cnames"], N_lineages, "R.")
#     med, lQ, uQ = get_quants(R, ps)

#     sim_freq = sample_posterior(states_dict[state]["stan_results"], states_dict[state]["stan_cnames"], N_lineages, "sim_freq")
#     freq_med, _, _ = get_quants(sim_freq, ps)

#     dates_vec = [] 
#     state_vec = []
#     lineage_vec = []
#     rt_median = []
#     rt_lower = [[] for i in 1:length(ps)]
#     rt_upper = [[] for i in 1:length(ps)]
#     freq_median = []

#     # May have to adjust to the size of things
#     seed_L = states_dict[state]["stan_data"]["seed_L"]
#     for lineage in 1:N_lineages
#         dates_vec = vcat(dates_vec, states_dict[state]["date"])
#         state_vec = vcat(state_vec, repeat([state], length(states_dict[state]["date"])))
#         lineage_vec = vcat(lineage_vec, repeat([seq_labels[lineage]], length(states_dict[state]["date"])))
#         rt_median = vcat(rt_median, med[:, lineage])
#         freq_median = vcat(freq_median, freq_med[:, lineage])
#         for i in 1:length(ps)
#             rt_lower[i] = vcat(rt_lower[i], lQ[i][:, lineage]) # Each col different p
#             rt_upper[i] = vcat(rt_upper[i], uQ[i][:, lineage])
#         end
#     end
    
#     rt_lower = vcat(rt_lower...)
#     rt_upper = vcat(rt_upper...)

#     return DataFrame(state = state_vec, 
#         date = dates_vec,
#         lineage = lineage_vec,
#         rt_median = rt_median,
#         rt_lower_50 = rt_lower[1],
#         rt_upper_50 = rt_upper[1],
#         rt_lower_80 = rt_lower[2],
#         rt_upper_80 = rt_upper[2],
#         rt_lower_95 = rt_lower[3],
#         rt_upper_95 = rt_upper[3],
#         freq_median = freq_median)
# end

# function get_growth_advantages_HDI(SoI, states_dict)
#     v = get_posterior(states_dict, SoI, "v", false)
#     v = [exp.(vi) for vi in v]
#     med, lQ, uQ = get_quants(v, ps)

#     state_vec = repeat([SoI], length(med))
#     lineage_vec =  vcat(states_dict[SoI]["seq_labels"]...)[1:end-1]

#     return DataFrame(state = state_vec, 
#                lineage = lineage_vec,
#                v_median = vec(med),
#                v_lower_50 = lQ[1],
#                v_upper_50 = uQ[1],
#                v_lower_80 = lQ[2],
#                v_upper_80 = uQ[2],
#                v_lower_95 = lQ[3],
#                v_upper_95 = uQ[3]
#                )    
# end
