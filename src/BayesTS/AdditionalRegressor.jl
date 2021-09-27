struct AdditionalRegressor <: RegressorNode
    regressor::Array{V, 2} where {V <: Real}
    n_regressors::T where {T <: Int}
end

function AdditionalRegressor(regressor::Array{T, 2}) where {T <: Real} 
    return AdditionalRegressor(regressor, size(regressor, 2))
end

params(TS::AdditionalRegressor) = (TS.regressor, TS.n_regressors)
params_shape(TS::AdditionalRegressor) = TS.n_regressors
get_priors(TS::AdditionalRegressor) = fill(Distributions.Normal(0.0, 10.0), params_shape(TS))

function (TS::AdditionalRegressor)(β, t)
    return TS.regressor * β
end

get_design(TS::AdditionalRegressor, t) = TS.regressor
