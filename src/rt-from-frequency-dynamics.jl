module rt_from_frequency_dynamics
    using CSV, Dates, DataFrames
    using Distributions
    using StanSample
    using CairoMakie

    import Base: sum, +, getindex # For Bayes TS
    import StatsBase: params
    
    export TimeSeriesNode
    export SplineTrend, LinearTrend, ConstantTrend, FlatTrend
    export FourierSeasonality
    
    include("BayesTS/BayesTS.jl") # Take BayesTS stuff and move into this package
    using .BayesTS

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

    export format_sequence_names, counts_to_matrix, prep_sequence_counts, prep_cases
    include("DataHelpers.jl")

    # Once interface is working
    # Trash: define_stan_model, process_state_data_for_stan,
    # Trash: Process all states, run_stan_model
    # Trash: run_all_stan_models
    # Clean: export get_posterior
    #include("InferenceHelpers.jl")

    export AggregateData, LineageData
    export SingleRModel, FreeLineageModel, FixedLineageModel
    export make_stan_data, make_stan_model, make_stan
    export run!, load_samples!

    # New 
    include("ModelData.jl")
    include("RModel.jl")
    include("StanPriors.jl")
    include("LineageModels.jl")
    include("ModelStan.jl")
    # Eventually: include("MLR.jl")

    # PosteriorHelpers is fine
    export sample_posterior, get_posterior, get_quants, hpd
    include("PosteriorHelpers.jl")

    export get_Rt_dataframe, get_growth_advantages_dataframe
    include("ExportData.jl")

    # TODO: Make growth advantage plots
    export lineage_to_WHO, lineage_colors, alphas, ps
    export get_nice_ticks, unpack_data, get_sequence_map
    export add_monthly_dates! #, make_lineage_legend!
    export plot_cases!, plot_observed_frequencies!
    export plot_lineage_R!, plot_average_R!, plot_lineage_R_censored!
    export plot_lineage_frequency!, plot_frequency_ppc!
    export plot_lineage_prev!, plot_smoothed_EC!
    include("MakiePlots.jl")
end
