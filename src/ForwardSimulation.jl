NegativeBinomial2(μ, α) = NegativeBinomial(inv(α),  inv(α)/(inv(α) + μ);  check_args=false)

function get_infections(R, I0, g_rev, seed_L)
    T = length(R) + seed_L
    l = length(g_rev)
    I = zeros(T)
    I[1:seed_L] .+= I0
    for t in seed_L:(l-1)
        I[t+1] = R[t-seed_L+1] * sum(I[1:t] .* g_rev[1:t])
    end
    for t in l:(T-1)
        I[t+1] = R[t-seed_L+1] * sum(I[(t-l+1):t] .* g_rev)
    end
    return I
end

function apply_delay(I, delay_rev)
    I_delay = similar(I)
    l = length(delay_rev)
    T = length(I)

    for t in 1:(l-1)
        I_delay[t] = sum(I[1:t] .* delay_rev[1:t])
    end
    for t in l:T
        I_delay[t] = sum(I[(t-l+1):t] .* delay_rev)
    end
    return I_delay
end

function apply_delays(I, delay_revs)
    I_ = copy(I)
    for d in eachcol(delay_revs)
        I_ = apply_delay(I_, d)
    end
    return I_
end

function get_cases_rng(I_prev, α, ρ)
    return rand.(NegativeBinomial2.(ρ .* I_prev, α))
end

function forward_simulate_I(R::Vector, I0, g_rev, delays_rev, seed_L)
    I = similar(R)
    I_delay = similar(R)

    I = get_infections(R, I0, g_rev, seed_L)
    I_delay = apply_delays(I, delays_rev)
    return I_delay
end

function forward_simulate_I(R::Matrix, I0, g_rev, delays_rev, seed_L)
    N_lineage = size(R, 2)
    run_lineage(k) = forward_simulate_I(R[:, k], I0[k], g_rev, delays_rev, seed_L) 
    return reduce(hcat, [run_lineage(lineage) for lineage in 1:N_lineage])
end

function get_reporting_vector(ρ, T)
    l = length(ρ)
    n_repeat = round(Int, T // l) + 1
    return repeat(ρ, n_repeat)[1:T]
end

function forward_simulate_C(R::Vector, I0, g_rev, delays_rev, α, ρ, seed_L)
    I = forward_simulate_I(R, I0, g_rev, delays_rev, seed_L)

    # Multinomial sampling
    obs_counts = sample_lineages(N, freq)

    # cases sampling
    obs_cases = get_cases_rng(total_I, α, ρ)
end

function sample_lineages(N, freq)
    T = length(N)
    obs_count = zeros(size(freq))

    for t in 1:T
        if N[t] > 0
            obs_count[t, :] += rand(Multinomial(N[t], freq[t,:]))
        end
    end
    return obs_count
end

# Add method to simulate cases and frequencies
function forward_simulate_lineages(R::Matrix, I0, g_rev, delays_rev, α, ρ, N, seed_L)
    I = forward_simulate_I(R, I0, g_rev, delays_rev, seed_L)

    total_I = sum(I, dims=2)
freq = (I ./ total_I)[(seed_L+1):end, :]

    # Multinomial sampling
    obs_counts = sample_lineages(N, freq)

    # cases sampling
    obs_cases = get_cases_rng(total_I[(seed_L+1):end], α, ρ)
    return obs_counts, obs_cases, I, freq
end

# Need two forward simulate. One N_deme, One single for arbitrary delays
# Add forward_simulate
# R: Matrix, I0, g, delays: Matrix each row new delay, α, ρ (vector), seed_L?
# And N: Counts each day
