abstract type RModel end

mutable struct SingleRModel{T <: Real} <: RModel
    g::Vector{T}
    delay::Vector{T}
    rt_model::TimeSeriesNode
    priors::Vector{V} where {V <: Any}
    seed_L::Int
    forecast_L::Int
end

function make_stan_data(model::SingleRModel, data::AggregateData)
    L = length(cases)
    obs_period = 1:L
    
    t = collect(obs_period)
    X = get_design(model.rt_model, t)
    
    stan_data = Dict(
        "seed_L" => model.seed_L,
        "forecast_L" => model.forecast_L,
        "L" => L,
        "cases" => data.cases,
        "g" => model.g,
        "onset" => model.delay, 
        "l" => length(model.g), # Will have to change to lg, lo
        "K" => size(X, 2),
        "features" => X
    )
    return stan_data
end

function make_stan_model(model::SingleRModel, model_name::String, tmpdir)
    # Update model with user priors and parameterizations 
    model_string = read(joinpath(@__DIR__, "stan_models/Rt_Structured_single.stan"), String); 
    model_string = add_R_priors_single(model_string, model.rt_model, model.priors) # These methods depend on what kind of time series model we're using?
    model_string = add_other_parms(model_string, model.rt_model, model.priors)
    return make_stan_model(model_string, model_name, tmpdir)
end