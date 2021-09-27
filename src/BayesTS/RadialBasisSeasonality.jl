struct RadialBasisSeasonality <: BasisNode
    Scale::V where {V <: Real}
    Period::V where {V <: Real}
    s::Vector{V} where {V <: Real}
end

## How do we want to do this?

params(TS::RadialBasisSeasonality) = (TS.Scale, TS.Period)
params_shape(TS::RadialBasisSeasonality) = 
