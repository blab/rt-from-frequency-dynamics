abstract type ModelData end

struct AggregateData <: Model
    cases::Vector{Int}
    dates::Vector{Dates}
end

function AggregateData(df::DataFrame) 
    return AggregateData(df.cases, df.date)
end

struct LineageData <: ModelData
    sequence_counts::Matrix{Int}
    cases::Vector{Int}
    seq_names::Vector{String}
    dates::Vector{Dates}
end

function LineageData(df::DataFrame)
    seq_names, seq_counts = sequence_counts_to_matrix(df)
    cases = df[:, :cases]
    dates = df[:, :date]
    return LineageData(seq_counts, cases, seq_names, dates)
end