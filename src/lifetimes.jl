# Edit to use Gamma and parameterize in terms of mean and sd
# I have code for this somewhere

function _discretize_dist(D, max_q)
    max_n = Int(round(quantile(D, max_q))) + 1
    xsd = 1:max_n
    xsd = vcat(0., collect(xsd .+ 0.5)) # 0.0, 1.5, 2.5, ..., max_n + 0.5
    return diff(cdf.(D, xsd)) # \int_{n-0.5}^{n+0.5} d(t) dt
end

function discrete_Gamma(μ, σ; max_q = 0.999)
    #μ = a*b
    #σ = sqrt(a) * b
    a = (μ / σ)^2
    b_inv = σ^2  / μ # b = (μ / σ^2)
    g = Gamma(a, b_inv) # k = a, θ = b^-1
    return _discretize_dist(g, max_q)
end

function discrete_LogNormal(μ, σ; max_q = 0.999)
    γ = 1 + σ^2 / μ^2
    LN = LogNormal(log(μ / sqrt(γ)), sqrt(log(γ)))
    return _discretize_dist(LN, max_q)
end

"""
Serial interval of novel coronavirus (COVID-19) infections
Nishiura. T
The author's fit the serial interval using a log-normal distribution.
"""
function SI_Nishiura()
    # Generate best fit log-norm to serial interval (Nishiura 2020)
    mean_serial = 4.7
    sd_serial = 2.9
    return discrete_LogNormal(mean_serial, sd_serial)    
end

"""
Generate onset time from distribution and maximum probability on symptom onset.
"""
function onset_time(dist, onset_max)
    return onset_max * dist / maximum(dist) 
end

# function onset_time_Gamma(m_g, std_g, onset_max)
#     dist = discrete_Gamma(m_g, std_g)
#     return onset_time(dist, onset_max)
# end

function get_standard_delays(; max_q = 0.999)
    # Ganyani 2020 for serial interval
    g = discrete_Gamma(5.2, 1.72, max_q=max_q)

    # Cheng 2021 for inclubation / onset
    onset = onset_time(discrete_LogNormal(6.9, 2.0, max_q=max_q), 1.0)
    return g, onset
end