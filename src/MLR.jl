mutable struct MLRData <: ModelData
    sequence_counts::Matrix{Int}
    seq_names::Vector{String}
    dates::Vector{Date}
end

function MLRData(seq_df::DataFrame)
    seq_names, dates, seq_counts = prep_sequence_counts(seq_df)
    return LineageData(seq_counts, seq_names, dates)
end

function make_stan_data(model::MLRModel, data::MLRData)
    N, N_variant = size(data.sequence_counts)
    stan_data = Dict(
        "N" => N,
        "D" => N_variant,
        "num_sequenced" => data.sequence_counts,
        "K" => size(model.features, 2),
        "X" => model.features
    )
    return stan_data
end

mutable struct MLRModel{T <: Real} <: LineageModel
    features::Matrix
end

function MLRModel(max_T) where {T <: Real}
    features = hcat(ones(max_T), collect(1:max_T))
    return MLRModel(features)
end


function make_stan_model(LM::MLR, model_name::String, tmpdir)
    model_string = read(joinpath(@__DIR__, "stan_models/MLR.stan"), String); 
    return make_stan_model(model_string, model_name, tmpdir)
end
