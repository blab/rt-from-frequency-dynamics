struct LinearTrend <: TrendNode  
    N_change::T where {T <: Int}
    s::Vector{V} where {V <: Real}
end

function LinearTrend(N_change::T, max_T::V) where {T <: Int, V <: Real} 
    return LinearTrend(N_change, Vector(LinRange(0, max_T, N_change+2))[2:end-1])
end

LinearTrend(s::Vector{T}) where {T <: Real} = LinearTrend(length(s), s)

params(TS::LinearTrend) = (TS.N_change, TS.s)
params_shape(TS::LinearTrend) = Int(TS.N_change)
get_components(TS::LinearTrend) = TS
get_priors(TS::LinearTrend) = [Normal(0.0, 5.0)]

function get_design(TS::LinearTrend, t)
    X = reduce(hcat, (t .- TS.s[k]) .* (t .> TS.s[k]) for k in 1:length(TS.s))
    return hcat(t, X) ./ TS.s[end]
end

function (TS::LinearTrend)(δ, r0, t)
    return get_design(TS, t) * vcat(r0, δ)
end

function (TS::LinearTrend)(θ, t) 
    return get_design(TS, t) * θ
end

# function make_A(t, s)
#     T, k = length(t), length(s)
#     A = zeros(T, k)
    
#     for j in 1:k
#         A[:, j] = t .> s[j]
#     end
    
#     return A
# end


## In Prophet, linear trend is given by
## (k + Aδ)t + (m + A(-sδ)),
## Where k is initial growth and (m + A(-s δ)) is the offset
## Here, we use δ as the intitial growth and the other delta represent the piecewise growth rate

# function make_piecewise_lin_trend(s, t, δ, r0, m)
#     # Add functionality to compute A once and a m
#     A = make_A(t, s)
#     return  m .+ (r0 .+ A*δ).*t - A*(s.*δ)
# end

struct ConstantTrend <: TrendNode  
    N_change::T where {T <: Int}
    s::Vector{V} where {V <: Real}
end

function ConstantTrend(N_change::T, max_T::V) where {T <: Int, V <: Real} 
    return ConstantTrend(N_change, Vector(LinRange(0, max_T, N_change+2))[2:end-1])
end

ConstantTrend(s::Vector{T}) where {T <: Real} = ConstantTrend(length(s), s)

params(TS::ConstantTrend) = (TS.N_change, TS.s)
params_shape(TS::ConstantTrend) = Int(TS.N_change)
get_components(TS::ConstantTrend) = TS
get_priors(TS::ConstantTrend) = [Normal(0.0, 5.0)]

function get_design(TS::ConstantTrend, t)
     reduce(hcat, t .> TS.s[k] for k in 1:length(TS.s))
end

function (TS::ConstantTrend)(θ, t)
    return get_design(TS, t) * θ
end

# function make_constant_trend(s, t, δ, m)
#     # Add functionality to compute A once and a m
#     A = make_A(t, s)
#     return  m .+ A * δ
# end

struct FlatTrend <: TrendNode end

params(TS::FlatTrend) = nothing
params_shape(TS::FlatTrend) = 1
get_priors(TS::FlatTrend) = [Normal(0.0, 20.0)]

function (TS::FlatTrend)(θ, t)
    get_design(TS, t) * θ
end

function get_design(TS::FlatTrend, t)
    return ones(eltype(t), size(t))
end

struct LogLinearTrend <: TrendNode  
    N_change::T where {T <: Int}
    s::Vector{V} where {V <: Real}
end

function LogLinearTrend(N_change::T, max_T::V) where {T <: Int, V <: Real} 
    return LogLinearTrend(N_change, Vector(LinRange(0, max_T, N_change+2))[2:end-1])
end

LogLinearTrend(s::Vector{T}) where {T <: Real} = LogLinearTrend(length(s), s)

params(TS::LogLinearTrend) = (TS.N_change, TS.s)
params_shape(TS::LogLinearTrend) = Int(TS.N_change)
get_components(TS::LogLinearTrend) = TS
get_priors(TS::LogLinearTrend) = [Normal(0.0, 5.0)]

log_where_positive(x) = @. ifelse(x > 0., log(x), 0.) 

function get_design(TS::LogLinearTrend, t)
    X = reduce(hcat, (t .- TS.s[k]) .* (t .> TS.s[k]) for k in 1:length(TS.s))
    return log_where_positive(hcat(t, X) ./ TS.s[end])
end

function (TS::LogLinearTrend)(δ, r0, t)
    return get_design(TS, t) * vcat(r0, δ)
end

function (TS::LogLinearTrend)(θ, t) 
    return get_design(TS, t) * θ
end

