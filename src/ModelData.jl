abstract type ModelData end

struct AggregateData <: ModelData
    cases::Vector{Int}
    dates::Vector{Date}
end

function AggregateData(df::DataFrame) 
    return AggregateData(df.cases, df.date)
end

struct LineageData <: ModelData
    sequence_counts::Matrix{Int}
    cases::Vector{Int}
    seq_names::Vector{String}
    dates::Vector{Date}
end

function LineageData(df::DataFrame)
    seq_names, seq_counts = sequence_counts_to_matrix(df)
    seq_names = [split(name, "_")[2] for name in seq_names]
    cases = df[:, :cases]
    dates = df[:, :date]
    return LineageData(seq_counts, cases, seq_names, dates)
end