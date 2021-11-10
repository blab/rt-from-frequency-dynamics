function define_stan_model(state, filename; 
    num_warmup = 2000, 
    num_samples = 1000,
    thin = 2,
    nchains = 4)
    # Construct Model 
     model_string = read(filename, String);
     stan_model = Stanmodel(
     thin = 2, 
     name = "rt-lineages-" * state,   
     nchains = 4,
     num_warmup = 2000, 
     num_samples = 1000,
     tmpdir = cd(pwd, "..") * "/data/sims/rt-" * state,
     model = model_string); 
 end
 
 function process_state_data_for_stan(state, df, g, onset, seed_L, forecast_L, k)
    state_df = load_state_data(state, df)
    seq_cols, seq_counts, seq_total = sequence_counts_to_matrix(state_df)
    cases = cases_to_vector(state_df)
    
    stan_data = get_data_for_inference(seed_L, forecast_L,
                                     cases, seq_counts,  seq_total, 
                                     g, onset, k)
    
    return state_df[:, :date], seq_cols, stan_data
end

function process_all_states(filename, df, g, onset, seed_L, forecast_L, k; 
    states_names = unique(df[: , :state]),
    num_warmup = 2000, 
    num_samples = 1000,
    thin = 2,
    nchains = 4)

    states_dict = Dict()
    
    for state in state_names
        dates_vec, seq_labels, state_data = process_state_data_for_stan(state, df, g, onset, seed_L, forecast_L, k)
        state = replace(state, ' ' => '_')
        state_model = define_stan_model(state, filename,
            num_warmup=num_warmup,
            num_samples=num_samples,
            thin=thin,
            nchains=nchains)
        
       states_dict[state] = Dict(
        "date" => dates_vec,
        "seq_labels" => clean_labels(seq_labels, state_data),
        "stan_data" => state_data,
        "stan_model" => state_model
        )
    end
    
    return states_dict
end

function run_stan_model!(state, states_dict)
    rc, sim, cnames = stan(states_dict[state]["stan_model"], states_dict[state]["stan_data"])
    
    states_dict[state]["rc"] = rc
    states_dict[state]["stan_results"] = sim
    states_dict[state]["stan_cnames"] = cnames
end

function run_all_stan_models!(states_dict; states_to_run = keys(states_dict), rerun = false)
    
    for state in states_to_run
        # Has this been run before?
        if haskey(states_dict[state], "rc") 
            # Are we re-running because of failure or user choice
            if states_dict[state]["rc"] != 0  || rerun # If there was an error, run again
                println("Running Stan on $state")
                run_stan_model!(state, states_dict)
            end
        else #!haskey(states_dict[state], "rc") # It has never run
            println("Running Stan on $state")
            run_stan_model!(state, states_dict)
        end
    end
end

function load_state_samples!(state, states_dict)
    model = states_dict[state]["stan_model"]
    sim, cnames = CmdStan.read_samples(model, false, false)
    
    states_dict[state]["stan_results"] = sim;
    states_dict[state]["stan_cnames"] = cnames;
end

function load_all_samples!(states_dict; states_to_run = keys(states_dict))
    
    for state in states_to_run
        # Has this been run before?
        if haskey(states_dict[state], "stan_results") 
            continue
        else 
            println("Loading $state samples")
            load_state_samples!(state, states_dict)
        end
    end
end

function get_posterior(states_dict, SoI, var, multi)
    if multi
        sample_posterior(states_dict[SoI]["stan_results"], 
                    states_dict[SoI]["stan_cnames"],
                    states_dict[SoI]["stan_data"]["N_lineage"], var)
    else
        sample_posterior(states_dict[SoI]["stan_results"], 
                    states_dict[SoI]["stan_cnames"], var)
    end
end
