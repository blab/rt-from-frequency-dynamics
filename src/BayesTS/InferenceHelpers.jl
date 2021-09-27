# """
# Takes in a TS and vector of parameters θ. 
# Returns reshaped θ_unpacked.
# """
# function unpack_parms(TS::TimeSeriesNode, θ)

#     p_shapes = params_shape(TS)

#     current_index = 1
#     new_index = current_index - 1

#     θ_unpacked = []

#     for c in 1:length(p_shapes)
#         p_num = prod(p_shapes[c])
#         new_index +=  p_num

#         push!(θ_unpacked,
#         reshape(θ[current_index:new_index], p_shapes[c]))

#         current_index += p_num
#     end

#     # Lastly, add σ
#     push!(θ_unpacked, θ[end])
#     return θ_unpacked
# end

# """
# Draw `num_samples` samples from a MCMC chain `chain` for use with `TimeSeriesNode`.
# """
# function sample_chain(TS::TimeSeriesNode, chain, num_samples)

#     θ = Array(sample(chain, num_samples))  

#     θ_unpacked = Array{Any}(undef, num_samples, n_components(TS) + 1)

#     for sample in 1:num_samples
#         θ_unpacked[sample, :] = unpack_parms(TS, θ[sample, :])
#     end

#     return θ_unpacked
# end

# """
# Draw from the predicive distribution given unpacked parameters `theta_unpacked`.
# """
# function sample_predictive(TS::TimeSeriesNode, t, θ_unpacked)
#     num_samples = size(θ_unpacked)[1]

#     predictive = Matrix{Float64}(undef, length(t), num_samples)

#     for sample in 1:num_samples
#         predictive[:, sample] = rand.(Normal.(TS(θ_unpacked[sample, 1:(end-1)], t), θ_unpacked[sample, end]))
#     end

#     return predictive
# end

# """
# Draw `num_samples` samples from the predictive distribution given a chain `chain`.
# """
# function sample_predictive(TS::TimeSeriesNode, t, chain, num_samples)
#     θ_unpacked = sample_chain(TS, chain, num_samples)
#     return sample_predictive(TS, t, θ_unpacked) 
# end

# function sample_time_series(t, N_change, F_nodes, Ps) 
#     @assert length(F_nodes) == length(Ps)
#     N_seas = length(F_nodes)
    
#     s = LinRange(0, maximum(t), N_change)[2:end]
#     δ = cumsum( 0.05 .* randn(N_change))
#     m = randn()
    
#     trend = make_trend(δ, s, m, t)

#     sampled_TS = trend
#     for i in 1:N_seas
#         sampled_TS += fourier_sum(t, randn(F_nodes[i], 2), Ps[i]) 
#     end

#     return sampled_TS
# end
#
#

function get_posterior_beta(samples, cnames)
    beta_idx = startswith.( cnames, "b")
    beta_samples = samples[:, beta_idx, :]
    long_beta = vcat([beta_samples[:, :, i] for i in 1:size(samples,3)]...) # Need to change for n.chains
    return transpose(long_beta)
end

function sample_posterior_predictive(model, t, samples, cnames, N_samples)
    β = get_posterior_beta(samples, cnames)
    sampled_cols = rand(1:size(β, 2), N_samples)
    β_sample = β[:, sampled_cols]
    
    X = get_design(model, t)
    return X * β_sample
end
