# Edit to use Gamma and parameterize in terms of mean and sd
# I have code for this somewhere

"""
Serial interval of novel coronavirus (COVID-19) infections
Nishiura. T
The author's fit the serial interval using a log-normal distribution.
"""
function generation_time(n_days = 20)
    # Generate best fit log-norm to serial interval (Nishiura 2020)
    mean_serial = 4.7
    sd_serial = 2.9
    
    γ = 1 + sd_serial^2 / mean_serial^2
    dist = LogNormal(log(mean_serial / sqrt(γ)), sqrt(log(γ)))
    # What are the breakpoints
    # 0.5 - 1.5, 1.5 - 2.5
    gen = cdf(dist, pushfirst!([day + 0.5 for day in 1:n_days], 0.0))
    gen = diff(gen)
    gen /= sum(gen)
end

"""
Get discretized generation time based on gamma distribution with given mode and std
"""
function generation_time_Gamma(n_days, m_g, std_g)
    ra = ( m_g + sqrt( m_g^2 + 4*std_g^2 ) ) / ( 2 * std_g^2 )
    sh = 1 + m_g * ra
    g = Gamma(sh, inv(ra))
    xsd = 1:n_days
    return diff(cdf.(g, vcat(0., collect(xsd .+ 0.5))))
end

"""
Generate generation time based on shape and rate parameters GS and GP.
"""
function generation_time(n_days, GP, GS)
    # Generate generation time based on parameters GP and GS

    x = pushfirst!([day + 0.5 for day in 1:n_days], 1e-16)
    
    gen = cdf(Normal(), (log.(x) .- log(GP)) * inv(GS))
    gen = diff(gen)
    gen /= sum(gen)
end


"""
Generate onset time based on shape and rate parameters GS and GP.
"""
function onset_time(n_days, OP, OS, onset_max)
    
    x = pushfirst!([day + 0.5 for day in 1:n_days], 1e-16)
    
    onset = cdf(Normal(), (log.(x) .- log(OP)) * inv(OS))
    onset = diff(onset)
    onset /= maximum(onset)
    onset_max * onset
end
