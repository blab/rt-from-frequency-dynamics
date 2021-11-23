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

# # Update and test OR choose to delete
# function LineageData(df::DataFrame)
#     seq_names, seq_counts = sequence_counts_to_matrix(df)
#     seq_names = [split(name, "_")[2] for name in seq_names]
# cases = df[:, :cases]
#     dates = df[:, :date]
#     return LineageData(seq_counts, cases, seq_names, dates)
# end

function LineageData(seq_df::DataFrame, cases_df::DataFrame)
    seq_names, dates_s, seq_counts = prep_sequence_counts(seq_df)
    dates_c, cases = prep_cases(cases_df)
    
    # Find overlap of dates to be safe
    idx_c = findall(d -> d in dates_s, dates_c)
    idx_s = findall(d -> d in dates_c, dates_s)
    
    return LineageData(seq_counts[idx_s, :], cases[idx_c], seq_names, dates_s[idx_s])
end
