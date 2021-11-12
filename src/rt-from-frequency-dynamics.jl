module rt_from_frequency_dynamics
    using CSV, Dates, DataFrame
    using Distributions
    using StanSample
    
    include("BayesTS/BayesTS.jl")
    using BayesTS

    export discrete_Gamma, discrete_LogNormal
    export onset_time, get_standard_delays
    
    export NegativeBinomial2

    include("lifetimes.jl")
    include("ForwardSimulation.jl")

    # include("BayesTShelpers.jl")
    # include("BayesTShelpersMulti.jl")

    # Once interface is working
    # Trash: get_data_obs_period, get_data_for_inference
    # Trash: process_state_data_for_stan, define_stan_model
    # Trash: Process all states -> (Make sure names and labels are cleaned)
    # Clean: export get_Rt_by_state_HDI, get_growth_advantages_HDI

    include("DataHelpers.jl")

    # Once interface is working
    # Trash: define_stan_model, process_state_data_for_stan,
    # Trash: Process all states, run_stan_model
    # Trash: run_all_stan_models
    # Clean: Load all samples -> load_samples.()
    # export load_samples
    # Clean: export_get_posterior
    include("InferenceHelpers.jl")


    # New 
    include("ModelData.jl")
    include("RModel.jl")
    include("StanPriors.jl")
    include("LineageModels.jl")
    include("ModelStan.jl")
    # Eventually: include("MLR.jl)

    # PosteriorHelpers is fine
    include("PosteriorHelpers.jl")

    # Refactor EVERYTHING here, make more flexible using Makie, so you provide ax, and colors
    include("PlotFunctions.jl")
end