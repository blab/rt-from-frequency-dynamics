struct RandomWalk
    init_dist::Distribution
    step_dist
    step_noise
end

function get_prior_string(TS::TimeSeriesNode, priors, s_idx::Int, e_idx::Int)
    n_dists = length(priors)

    # If one distribution is provided, provide to all arguments
    if n_dists == 1 && n_dists != params_shape(TS)
        #prior_tmp = "to_vector(b[$s_idx : $e_idx]) ~ PRIOR_TEXT; \n // ADD_MORE_PRIORS \n" 
        prior_tmp = "b[$(s_idx):$(e_idx)] ~ PRIOR_TEXT; \n // ADD_MORE_PRIORS \n" 
        prior_str = replace(prior_tmp, "PRIOR_TEXT" => stan_string(priors[1]))
    end

    if n_dists == params_shape(TS) || (s_idx == e_idx)
        
        prior_tmp = "b[IDX_TEST] ~ PRIOR_TEXT; \n"
        
        prior_str = ""
        for i in 1:n_dists
            tmp = replace(prior_tmp, "IDX_TEST" => "$(s_idx+i-1)")
            prior_str *= replace(tmp, "PRIOR_TEXT" => stan_string(priors[i]))
        end

        prior_str *= "// ADD_MORE_PRIORS"
    end
    
    return prior_str
end

function get_prior_string(TS, s_idx, e_idx)
    get_prior_string(TS, get_priors(TS), s_idx::Int, e_idx::Int)
end

function get_prior_string(TS::TimeSeriesNode, prior::RandomWalk, s_idx::Int, e_idx::Int)
    str_prior = "
        b[$s_idx] ~ INIT_PRIOR_TEXT;
        for (i in $(s_idx):$(e_idx-1)){
            b[i+1] ~ STEP_PRIOR_TEXT(b[i], STEP_TAU);
        }\n

        // ADD_MORE_PRIORS
        " 
    
    prior_str = replace(str_prior, "INIT_PRIOR_TEXT" => stan_string(prior.init_dist))
    prior_str = replace(prior_str, "STEP_PRIOR_TEXT" => stan_string(prior.step_dist))
    prior_str = replace(prior_str, "STEP_TAU" => "$(prior.step_noise)")

    return prior_str
end

function add_priors(model_file, model, priors=nothing)
    if isnothing(priors)
        priors = [get_priors(model[i]) for i in 1:n_components(model)]
    end

    # Count indices in β
    current_index = 1
    new_index = current_index - 1

    for i in 1:n_components(model)
        new_index += params_shape(model[i])
        prior_str = get_prior_string(model[i], priors[i], current_index, new_index)
        model_file = replace(model_file, "// ADD_MORE_PRIORS" => prior_str)
        current_index += params_shape(model[i])
    end

    return model_file
end

function add_observation_model(model_file, obs_dist)
    if obs_dist == "normal" || isnothing(obs_dist)
        model_file = replace(model_file, "// ADD_Y_INIT" => "vector[T] Y;")
        model_file = replace(model_file, "// ADD_OBS_PARMS" => "real<lower=0> sigma;")
        model_file = replace(model_file, "// ADD_EY_TRANS" => "vector[T] EY = features * b;")
        model_file = replace(model_file, "// ADD_OBS_PRIOR" => "sigma ~ exponential(0.5);")
        model_file = replace(model_file, "// ADD_OBS_DIST" => "normal(EY, sigma);")
    end
    
    if obs_dist == "poisson"
        model_file = replace(model_file, "// ADD_Y_INIT" => "int Y[T];")
        model_file = replace(model_file, "// ADD_EY_TRANS" => "vector[T] EY = features * b;")
        model_file = replace(model_file, "// ADD_OBS_DIST" => "poisson_log(EY);")
    end
    
    return model_file
end
