struct FourierSeasonality <: SeasonalityNode
    F_Nodes::T where {T <: Int}
    Period::V where {V <: Real}
end

params(TS::FourierSeasonality) = (TS.F_Nodes, TS.Period)
params_shape(TS::FourierSeasonality) = 2 * TS.F_Nodes
get_priors(TS::FourierSeasonality) = [Normal(0.0, 10.0)] # for each index 

function (TS::FourierSeasonality)(β, t) 
    return get_design(TS, t) * β
end

function get_design(TS::FourierSeasonality, t)
    trig_args = reduce(hcat, 2 * π * n * t / TS.Period for n in 1:TS.F_Nodes)    
    return hcat(sin.(trig_args), cos.(trig_args))  
end
