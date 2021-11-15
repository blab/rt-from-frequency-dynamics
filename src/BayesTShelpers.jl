# To be removed!
function BayesTS.get_prior_string(TS::TimeSeriesNode, priors, s_idx::Int, e_idx::Int)
    n_dists = length(priors)
    if priors[1] == "LAS"
        prior_str = """zgam ~ uniform(0., 1.);
\nb[$(s_idx)] ~ normal(0.,1.); 
\nfor (t in $(s_idx):$(e_idx-1)){
  b[t+1] ~ double_exponential(b[t], gam);
}
\n// ADD_MORE_PRIORS \n""" 
        return prior_str
    end

    # If one distribution is provided, provide to all arguments
    if n_dists == 1 && n_dists != params_shape(TS)
        prior_tmp = "to_vector(b[$(s_idx):$(e_idx)]) ~ PRIOR_TEXT; \n // ADD_MORE_PRIORS \n" 
       prior_str = replace(prior_tmp, "PRIOR_TEXT" => BayesTS.stan_string(priors[1]))
    end

    if n_dists == params_shape(TS) || (s_idx == e_idx)
        
        prior_tmp = "b[IDX_TEST] ~ PRIOR_TEXT; \n"
  
        prior_str = ""
        for i in 1:n_dists
            tmp = replace(prior_tmp, "IDX_TEST" => "$(s_idx+i-1)")
            prior_str *= replace(tmp, "PRIOR_TEXT" => BayesTS.stan_string(priors[i]))
        end

        prior_str *= "// ADD_MORE_PRIORS"
    end
    
    return prior_str
end

function add_misc_string!(model_file, TS::TimeSeriesNode, priors, s_idx::Int, e_idx::Int)
    if priors[1] == "LAS"
        str_1 = "real<lower=0, upper=1> zgam; // Squared Global Variance
\n// ADD_MORE_PARMS \n"

str_2 = "real gam = 0.5 * tan(zgam * pi() / 2.); // Random Walk Variance
\n // ADD_MORE_TRANS_PARMS"
        
        model_file = replace(model_file, "// ADD_MORE_PARMS" => str_1)
        model_file = replace(model_file, "// ADD_MORE_TRANS_PARMS" => str_2)
    end
    return model_file
end

function add_other_parms(model_file, model, priors=nothing)
    if isnothing(priors)
        return model_file
    end
            
    # Count indices in β
    current_index = 1
    new_index = current_index - 1

    for i in 1:n_components(model)
        new_index += params_shape(model[i])
        model_file = add_misc_string!(model_file, model[i], priors[i], current_index, new_index)
        current_index += params_shape(model[i])
    end
    return model_file
end 
