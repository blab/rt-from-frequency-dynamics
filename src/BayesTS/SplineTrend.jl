struct SplineTrend <: TrendNode
    N_change::T where {T <: Int}
    s::Vector{V} where {V <: Real}
    order::T where {T <: Int}
end

function SplineTrend(N_change::T, max_T::V, order::T) where {T <: Int, V <: Real}
    return SplineTrend(N_change, Vector(LinRange(1.0, max_T, N_change)), order)
end

function SplineTrend(s::Vector{V}, order::T) where {T <: Int, V <: Real}
    SplineTrend(length(s), s, order)
end

params_shape(TS::SplineTrend) = TS.N_change + TS.order - 2
params(TS::SplineTrend) = (TS.N_change, TS.s, TS.order)
get_components(TS::SplineTrend) = TS

function (TS::SplineTrend)(θ, t) 
    return get_design(TS, t) * θ 
end

get_omega(s1, s2, t::Vector{T}) where {T <: Real}= (s1 == s2) ? zero(T) : (t .- s1) ./ (s2 - s1)

function make_B_splines(t, s, order, i)
    if order == 1        
        return (t .>= s[i]) .* (t .< s[i+1]) 
    else    
        ω1 = get_omega(s[i], s[i+order-1], t)
        B1 = make_B_splines(t, s, order-1, i)
        ω2 = get_omega(s[i+1], s[i+order], t)
        B2 = make_B_splines(t, s, order-1, i+1)
        return ω1 .* B1  + (1 .- ω2) .* B2
    end 
end

function make_B_splines(t, s, order)
    extend_knots = vcat(repeat([s[1]], order-1), s, repeat([s[end]], order-1))
    X = reduce(hcat, make_B_splines(t, extend_knots, order, i) for i in 1:(length(s) + order-2))
    X[end, end] = 1
    return X
end

function get_design(TS::SplineTrend, t)
    return make_B_splines(t, TS.s, TS.order)
end

