abstract type LineageModel <: RModel end

function make_stan_data(model::LineageModel, data::LineageData)
    L, N_lineage = size(data.sequence_counts)
    obs_period = 1:L
    
    t = collect(obs_period)
    X = get_design(model.rt_model, t)
    
    stan_data = Dict(
        "seed_L" => model.seed_L,
        "forecast_L" => model.forecast_L,
        "L" => L,
        "N_lineage" => N_lineage,
        "cases" => data.cases,
        "num_sequenced" => data.sequence_counts,
        "N_sequences" => Int.(vec(sum(data.sequence_counts, dims=2))),
        "g" => model.g,
        "onset" => model.delay, 
        "lg" => length(model.g),
        "lo" => length(model.delay),
        "K" => size(X, 2),
        "features" => X
    )
    return stan_data
end

mutable struct FreeLineageModel{T <: Real} <: LineageModel
    g::Vector{T}
    delay::Vector{T}
    rt_model::TimeSeriesNode
    priors::Vector{V} where {V <: Any}
    seed_L::Int
    forecast_L::Int
end

# Switching to StanSample -> This should go in InferenceHelpers
function make_stan_model(model_string::String, model_name::String, tmpdir)
    mkpath(abspath(tmpdir))
    stan_model = SampleModel(model_name, model_string, tmpdir = abspath(tmpdir))
    return stan_model
end

function make_stan_model(LM::FreeLineageModel, model_name::String, tmpdir)
    # Update model with user priors and parameterizations 
    model_string = read(joinpath(@__DIR__, "stan_models/Rt_Structured_lineage_ind.stan"), String); 
    model_string = add_R_priors_multi(model_string, LM.rt_model, LM.priors) # These methods depend on what kind of time series model we're using?
    model_string = add_other_parms(model_string, LM.rt_model, LM.priors)
    return make_stan_model(model_string, model_name, tmpdir)
end


mutable struct FixedLineageModel{T <: Real} <: LineageModel
    g::Vector{T}
    delay::Vector{T}
    rt_model::TimeSeriesNode
    priors::Vector{V} where {V <: Any}
    seed_L::Int
    forecast_L::Int
end

function make_stan_model(LM::FixedLineageModel, model_name::String, tmpdir)
    # Update model with user priors and parameterizations 
    model_string = read(joinpath(@__DIR__, "stan_models/Rt_Structured_lineage.stan"), String); 
    model_string = add_R_priors_single(model_string, LM.rt_model, LM.priors)
    model_string = add_other_parms(model_string, LM.rt_model, LM.priors)
    return make_stan_model(model_string, model_name, tmpdir)
end