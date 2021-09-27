abstract type TimeSeriesNode end
abstract type BasisNode <: TimeSeriesNode end

# BasisNodes can be represented as a design matrix
abstract type TrendNode <: BasisNode end # Trend Components
abstract type SeasonalityNode <: BasisNode end # Seasonality Components
abstract type RegressorNode <: BasisNode end # Additional Regressors

n_components(TS::TimeSeriesNode) = 1
get_components(TS::TimeSeriesNode) = [TS]
Base.getindex(TS::TimeSeriesNode, i) = (i==1) ? TS : nothing

function get_components(NodeL::TimeSeriesNode, NodeR::TimeSeriesNode)
    return vcat(get_components(NodeL), get_components(NodeR))
end

# Combining Nodes
# A lot of this archectiecture is shared. Could reduce to BinaryTSNode abstract type?

abstract type BinaryTSNode <: TimeSeriesNode end

params_shape(TS::BinaryTSNode) = hasproperty(TS, :Components) ? params_shape.(TS.Components) : nothing
n_components(TS::BinaryTSNode) = hasproperty(TS, :Components) ? length(TS.Components) : nothing
get_components(TS::BinaryTSNode) = hasproperty(TS, :Components) ? TS.Components : nothing
Base.getindex(TS::BinaryTSNode, i) = hasproperty(TS, :Components) ? TS.Components[i] : nothing

# Using param_shape and n_components I should be able to just have a function that
# calls the proper prior text based on the the indices
# Would this just be function like get_prior_string(TS, start, end) and you check if end - start is right length?

# function get_design(TS::BinaryTSNode, t, DesignType)
#     TypeComps = TS.Components[typeof.(TS.Components) .<: DesignType]
#    return reduce(hcat, [get_design(comp, t) for comp in TypeComps]) 
# end

# function get_design(TS::BinaryTSNode, t)
#     ComponentTypes = unique(typeof.(TS.Components))

#     DesignMats = []
#     for CType in ComponentTypes 
#         if CType <: BasisNode
#             push!(DesignMats, get_design(TS, t, CType))
#         end
#     end

#     return DesignMats
# end

function get_design(TS::BinaryTSNode, t)
    BasisComps = TS.Components[typeof.(TS.Components) .<: BasisNode]
   return reduce(hcat, [get_design(comp, t) for comp in BasisComps]) 
end

struct SumTSNode <: BinaryTSNode
    NodeL::TimeSeriesNode
    NodeR::TimeSeriesNode
    Components::Vector{TimeSeriesNode}

    SumTSNode(L, R) = new(L, R, get_components(L, R)) 
end

Base.:+(NodeL::TimeSeriesNode, NodeR::TimeSeriesNode) = SumTSNode(NodeL, NodeR)

# params_shape(TS::SumTSNodes) = params_shape.(TS.Components)
# n_components(TS::SumTSNodes) = length(TS.Components)
# get_components(TS::SumTSNodes) = TS.Components

function (TS::SumTSNode)(θ, t)
    # Need to Mark Sure θ is a vector of arrays with the right size for each comp
    # Get Params of Right Size for Each Component
   return sum(TS.Components[c](θ[c], t) for c in 1:n_components(TS))
end

struct ProdTSNode <: BinaryTSNode
    NodeL::TimeSeriesNode
    NodeR::TimeSeriesNode
    Components::Vector{TimeSeriesNode}

    ProdTSNode(L, R) = new(L, R, get_components(L, R)) 
end

Base.:*(NodeL::TimeSeriesNode, NodeR::TimeSeriesNode) = ProdTSNode(NodeL, NodeR)
## Need Add Mult, Fit 

## Example Code

## week_season = FourierSeasonality(3, 7.0)
## year_season = FourierSeasonality(10, 365.0)
## total_trend = LinearTrend(20.0)

## TimeSeriesModel = total_trend + year_season + week_season

## fit(TimeSeriesModel, t, data)

## This object needs to be able to save a chain, so it can be sampled from easily.
## Want two or three methods: post sample, prior sample, post-pred.


## When I call Structure with an Abstract Vector I want to evalute

## What option to make estiamte Hierarchical for various groups.
## Focus on a single time series first nerd


## Need method to return number of parameters


## Make Method for Shape and Size of params i.e. each Seasonality gets a β_? of size (F_nodes, 2)
