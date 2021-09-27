using Plots
using LinearAlgebra

# Observed times
t = collect(1:700)

# Number of change points
N_change = 10 

# Naive Vector of change points
s = LinRange(0, maximum(t), N_change)[2:end]

# What are growth rates
# First value of delta is initial growth
δ = cumsum( 0.01 .* randn(N_change+1))

# What is the intercept?
m = 0

trend = make_trend(δ, s, t, m)

plot(t, trend,
    label = false)

    ## Abstract Type TimeSeriesNode

    ## LinearTrend Inh from it

    ## Seasonality Inher from it

    ## Fourier is a concrete

"""
\beta is an array of Fourier coefficients.
"""
function fourier_sum(t, β, P)
    @assert size(β)[2] == 2
    
    N = size(β)[1]
    trig_args = hcat([2*π* n * t / P for n in 1:N]...)

    return sum( (β[:,1]' .* cos.(trig_args)) + (β[:,2]' .* sin.(trig_args)), dims = 2)
end


F_nodes = 2
week_season = fourier_sum(t, randn(F_nodes, 2), 7)
year_season = fourier_sum(t, randn(3*F_nodes, 2), 365.5) 

plot(t, trend + week_season + year_season,
    label = false,
    title = "Test Time Series ")

test_data = sample_time_series(t, 10, [3, 12], [7.0, 365.25]) 

plot(t, sample_time_series(t, 10, [3, 12], [7.0, 365.25]) ,
    label = false,
    title = "Test Time Series")

plot!(t, sample_time_series(t, 5, [3, 7], [7.0, 365.25]) ,
    label = false,
    title = "Test Time Series")

plot!(t, sample_time_series(t, 7, [3, 15], [7.0, 365.25]),
    label = false,
    title = "Test Time Series")

plot!(t, sample_time_series(t, 3, [3, 2], [7.0, 365.25]), 
    label = false,
    title = "Test Time Series")
