lineage_to_WHO = Dict()
lineage_to_WHO["B.1.1.7"] = "Alpha"
lineage_to_WHO["B.1.351"] = "Beta"
lineage_to_WHO["P.1"] = "Gamma"
lineage_to_WHO["B.1.617.2"] = "Delta"
lineage_to_WHO["B.1.525"] = "Eta"
lineage_to_WHO["B.1.526"] = "Iota "
lineage_to_WHO["B.1.617.1"] = "Kappa"
lineage_to_WHO["C.37"] = "Lambda"
lineage_to_WHO["other"] = "other"
lineage_to_WHO["B.1.427"] = "Epsilon"
lineage_to_WHO["B.1.621"] = "Mu"

lineage_colors = [ "#5081b9" "pink" "#ffdf64" "orange" "#40916c" "#ff6666" "#8a897c"]
lineage_colors = ["#2e5eaa", "#5adbff", "#f03a47", "#56e39f","#b4c5e4",  "#f5bb00", "#9e4244", "#808080"] 


alphas = [0.65, 0.45, 0.35]
ps = [0.5 0.8 0.95]
lQuants = 0.5 * (1. .- ps)
uQuants = 0.5 * (1. .+ ps)

check_if_first(d) = (Day(d) == Day(1)) ? true : false

function get_nice_ticks(days)
    ticks = findall(check_if_first, days)
    labels = Dates.monthabbr.(days[ticks]) 
   return ticks, labels 
end

function unpack_data(MS)
    dates_num = collect(1:MS.data["L"])
    seed_L = MS.data["seed_L"]
    forecast_L = MS.data["forecast_L"]
    N_lineage = MS.data["N_lineage"]
    return dates_num, seed_L, forecast_L, N_lineage
end

# lineage to index map
function get_sequence_map(seq_labels)
    lineage_map = Dict()
    for (i, lineage) in enumerate(sort(seq_labels))
        lineage_map[lineage] = i
    end
    return lineage_map
end

function plot_cases!(ax, MS; color = (:black, 0.3))
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    barplot!(ax, dates_num, MS.data["cases"], color = color) #, kwargs...)
end

function plot_observed_frequencies!(ax, MS; colors=lineage_colors; size = N -> 3)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    sample_freq = MS.data["num_sequenced"] ./ sum(MS.data["num_sequenced"], dims = 2)
    N_samples = vec(sum(MS.data["num_sequenced"], dims = 2))
    
    for lineage in 1:N_lineage
        scatter!(ax, dates_num, sample_freq[:,lineage],
            color = (colors[lineage], 1.0),
            strokewidth = 1.5,
            markersize = size.(N_samples))
    end
end

function plot_lineage_R!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    R = get_posterior(MS, "R.", true)
    med, lQ, uQ = get_quants(R, ps)

    # Horizontal dashed line to indicate 1.
    lines!(ax, dates_num, fill(1., length(dates_num)), color = "black", linestyle=:dash)  

    for lineage in 1:N_lineage
        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:, lineage], uQ[i][:, lineage], 
                color = (colors[lineage], alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end
        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", label = "Median")
    end
end

function plot_lineage_R_censored!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps, thres = 0.001)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    R = get_posterior(MS, "R.", true)
    med, lQ, uQ = get_quants(R, ps)

    sim_freq = get_posterior(MS, "sim_freq", true)
    med_freq, _, _ = get_quants(sim_freq, ps)
    
    # Horizontal dashed line to indicate 1.
    lines!(ax, dates_num, fill(1., length(dates_num)), color = "black", linestyle=:dash)  

    for lineage in 1:N_lineage
        # Which dates have posterior_median_frequency > 0.001
        censored = findall(x -> (x > thres), med_freq[:,lineage])
        
        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num[censored], 
                lQ[i][censored, lineage], uQ[i][censored, lineage], 
                color = (colors[lineage], alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end
        # Add median
        lines!(ax, dates_num[censored], med[censored,lineage], color = "black", label = "Median")
    end
end

function plot_average_R!(ax, MS; color=:purple, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    R_average = get_posterior(MS, "R_average.", false)
    med, lQ, uQ = get_quants(R_average, ps)
    
    hlines!(ax, [1.], color = "black", linestyle=:dash)  
    
    # Plot credible intervals
    for i in reverse(1:length(ps))
        band!(ax, dates_num, 
            lQ[i],  uQ[i], 
            color = (color, alphas[i]), 
            label = "$(Int(ps[i] * 100))% CI")
    end

    # Add median
    lines!(ax, dates_num, med, color = "black", linewidth = 1.5, label = "Median")
end

function plot_lineage_frequency!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    sim_freq = get_posterior(MS, "sim_freq", true)
    med, lQ, uQ = get_quants(sim_freq, ps)

    for lineage in 1:N_lineage
        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[i][:,lineage], 
                color = (colors[lineage], alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end
        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
    end
end

function plot_frequency_ppc!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    obs_freq = get_posterior(MS, "obs_freqs", true)
    med, lQ, uQ = get_quants(obs_freq, ps)

    for lineage in 1:N_lineage
        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[i][:,lineage], 
                color = (colors[lineage], alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
    end
end

function plot_lineage_prev!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    scaled_prev = get_posterior(MS, "scaled_prev.", true)
    med, lQ, uQ = get_quants(scaled_prev, ps)

    for lineage in 1:N_lineage
        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[i][:,lineage], 
                color = (colors[lineage], alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
    end
end

function plot_smoothed_EC!(ax, MS; color=:purple, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    
    EC_smooth = get_posterior(MS, "EC_smooth.", false)
    med, lQ, uQ = get_quants(EC_smooth, ps)

    # Plot credible intervals
    for i in reverse(1:length(ps))
        band!(ax, dates_num, 
            lQ[i],  uQ[i], 
            color = (color, alphas[i]), 
            label = "$(Int(ps[i] * 100))% CI")
    end
    
    # Add median
    lines!(ax, dates_num, med, color = "black", linewidth = 1.5, label = "Median")
end

function plot_growth_advantage!(ax, MS; colors=lineage_colors, alphas=alphas, ps=ps)
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    v = get_posterior(MS, "v", false)
    v = vcat(hcat(v...)')
    
    hlines!(ax, [1.0], color = :black, linestyle = :dash)
    for lineage in 1:(N_lineage-1)
        violin!(ax, fill(lineage, size(v, 1)), exp.(v[:, lineage]), 
            color = colors[lineage], 
            orientation=:vertical,
            width = 0.25,
            strokewidth = 1.5,
            strokecolor = :black)
        scatter!(ax, fill(lineage, size(v, 1)), exp.(v[:, lineage]), color = (:black, 0.1))
    end
    ax.xticks = 1:N_lineage
end

function add_monthly_dates!(ax, dates; skip=1)
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks[1:skip:end]
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
end
