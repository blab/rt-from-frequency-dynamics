function define_stan_model(TS::TimeSeriesNode, t; priors = nothing, num_samples = 2000, obs_dist = nothing)
    # How do we decide which file to use?

    # Load Base File and add Priors
    # Use joinpath and @__DIR__
    model_file = read("../src/stan/BayesTS.stan", String)
    model_file = add_priors(model_file, TS, priors)
    model_file = add_observation_model(model_file, obs_dist)

    stan_model = SampleModel("BayesTS", model_file;
    method = StanSample.Sample(
    save_warmup=false,
    thin=1,
    num_samples = num_samples
    )
    )

    return stan_model
end

function make_stan_data(TS::TimeSeriesNode, t, data)
    X = get_design(TS, t)

    return Dict(
    "T" => length(t),
    "t" => t,
    "Y" => data,
    "K" => size(X, 2),
    "features" => X)
end

function run_stan_model(stan_model, stan_data)
    rc = stan_sample(stan_model, data = stan_data, n_chains = 4)

    if success(rc)
        samples, cnames = read_samples(stan_model; output_format=:array, return_parameters=true)
    end

    return rc, samples, cnames
end

function sample_stan(
    TS::TimeSeriesNode,
    t,
    data;
    priors = nothing,
    num_samples = 5000,
    obs_dist = nothing)

    stan_model = define_stan_model(TS, t; priors = priors, num_samples = num_samples, obs_dist = obs_dist)
    stan_data = make_stan_data(TS, t, data)
    rc, samples, cnames = run_stan_model(stan_model, stan_data)
    return samples, cnames
end

# Add methods for posterior samples using stan

# Loading post 

# Saving post

# Plotting Components

# Adding forecast

