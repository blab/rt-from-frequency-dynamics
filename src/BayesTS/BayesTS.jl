module BayesTS

using StatsBase
using Distributions
using StanSample

import Base: sum, +, getindex
import StatsBase: params

export 
    # Types
    TimeSeriesNode,
    BinaryTSNode,
    SumTSNode,
    ProdTSNode,
    ## Individual Components
    FourierSeasonality,
    LinearTrend,
    ConstantTrend,
    FlatTrend,
    SplineTrend,
    ## Methods
    params, 
    params_shape,
    get_components,
    n_components,
    get_design,
    # Inference
    get_priors,
    get_prior_string,
    # sample_turing,
    add_priors,
    sample_stan,
    get_posterior_beta,
    sample_posterior_predictive

# Include Files
include("TimeSeriesNodes.jl")
include("LinearTrend.jl")
include("SplineTrend.jl")
include("FourierSeasonality.jl")
include("AdditionalRegressor.jl")
# include("TuringInference.jl")
include("stan_string.jl")
include("InferenceHelpers.jl")
include("StanUtilities.jl")
include("StanInference.jl")

end # module
