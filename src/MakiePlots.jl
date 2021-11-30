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

# function get_nice_ticks(days)
#     ticks = findall(check_if_first, days)
#     labels = Dates.monthabbr.(days[ticks]) 
#    return ticks, labels 
# end

function get_nice_ticks(days; get_past=true)
    ticks = findall(check_if_first, days)
    labels = Dates.monthabbr.(days[ticks]) 
    
    # If you want the label for last month before your data begins
    if (Dates.monthabbr(days[1]) != Dates.monthabbr(days[ticks[1]])) & get_past
       pushfirst!(ticks, 2 - Dates.dayofmonth(days[1]))
       pushfirst!(labels, Dates.monthabbr(days[1]))
    end
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

function plot_observed_frequencies!(ax, MS; colors=lineage_colors, size = N -> 3)
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
        if length(censored) > 0
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

# function add_monthly_dates!(ax, dates; skip=1)
#     ticks, _ = get_nice_ticks(dates)
#     ax.xticks = ticks[1:skip:end]
#     ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
# end

function add_monthly_dates!(ax, dates; skip=1, get_past=true)
    ticks, labels = get_nice_ticks(dates, get_past=get_past)
    tl = Dict(t => l for (t,l) in zip(ticks, labels)) 
    ax.xticks = ticks[1:skip:end]
    ax.xtickformat = xs -> [tl[convert.(Int, x)] for x in xs]
end


function make_plot_dataframe(obs_cases, obs_counts, lineage_names)
    cases = Int[]
    counts = Int[]
    obs_freqs = Float64[]
    dates = Int[]
    lineages = String[]
    lineages_num = Int[]
   # state = String[]
    
    T = length(obs_cases)
    N_lineage = size(obs_counts,2)
    
    obs_freq = obs_counts ./ sum(obs_counts, dims=2)
    
    
    for lineage in 1:N_lineage
        dates = vcat(dates, collect(1:T))
        cases = vcat(cases, obs_cases)
        counts = vcat(counts, obs_counts[:, lineage])
        obs_freqs = vcat(obs_freqs, obs_freq[:, lineage])
        lineages_num = vcat(lineages_num, repeat([lineage], T))
        lineages  = vcat(lineages, repeat([lineage_names[lineage]], T))
    #    state = vcat(state, repeat([state_name], T))
    end
    
    return DataFrame(date = dates, cases = cases, counts = Int.(counts), obs_freqs = obs_freqs, lineage = lineages, lineage_num = lineages_num) #, state = state)   
end

function figure_1(MS, LD, colors; figsize=(3200, 2600), fontsize=32, font="Helvetica")
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    seq_labels = LD.seq_names
    dates = LD.dates 

    lineage_map = get_sequence_map(seq_labels)
    cases = MS.data["cases"]
    counts =  MS.data["num_sequenced"]
    plot_data = make_plot_dataframe(cases, counts, seq_labels)
    color_vec = [colors[i] for i in plot_data.lineage_num]
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (3200, 2600), fontsize = fontsize, font = font)
 
    g_cases_rt =  fig[1:3,1:2] = GridLayout()
    
    ##################### CASES AND AGGREGATE SMOOTH ###############
    ax_cases = Axis(g_cases_rt[1:2,1], ylabel = "Observed Cases")
    plot_cases!(ax_cases, MS)
    plot_smoothed_EC!(ax_cases, MS)
    add_monthly_dates!(ax_cases, LD.dates)
    hidexdecorations!(ax_cases, grid = false)
    
    ###################### AVERAGE RT ##############################
    ax_average_rt = Axis(g_cases_rt[3,1], ylabel = L"R_{t}")
    plot_average_R!(ax_average_rt, MS)
    add_monthly_dates!(ax_average_rt, LD.dates)
    linkxaxes!(ax_cases, ax_average_rt)

    g_variant = fig[1:3, 3:4] = GridLayout()
    
    ###################### Lineage Prevalence ######################
    ax_smooth_lin = Axis(g_variant[1:2,1:2])
    plot_lineage_prev!(ax_smooth_lin, MS, colors = colors)
    add_monthly_dates!(ax_smooth_lin, LD.dates)
    hidexdecorations!(ax_smooth_lin, grid = false)
    
    ###################### Lineage Effective Reproductive Number ###
    ax_Rt = Axis(g_variant[3,1:2])
    plot_lineage_R_censored!(ax_Rt, MS, colors = colors)
    add_monthly_dates!(ax_Rt, LD.dates)
    linkxaxes!(ax_smooth_lin, ax_Rt)

    ##################### PLOTTING ORIGINAL SAMPLES ################
    g_seq_count = fig[4:5,1:4] = GridLayout()
    
    ax_obs_cases = Axis(g_seq_count[1:2,2], ylabel = "Observed Cases")
    plot_cases!(ax_obs_cases, MS)
    add_monthly_dates!(ax_obs_cases, LD.dates, skip = 2)
    
    ax_seq_count = Axis(g_seq_count[1,1], ylabel = "Observed Counts")
    barplot!(ax_seq_count, plot_data.date, plot_data.counts, 
        stack = plot_data.lineage_num,
        color = color_vec)    
    
    ##################### PLOTTING OBSERVED FREQ ################
    ax_seq_freq = Axis(g_seq_count[2,1], ylabel = "Sample Frequency")
    barplot!(ax_seq_freq, plot_data.date, plot_data.obs_freqs, 
        stack = plot_data.lineage_num,
        color = color_vec)    
    
    linkxaxes!(ax_seq_count, ax_seq_freq)
    hidexdecorations!(ax_seq_count, grid = false)
    
    add_monthly_dates!(ax_seq_freq, LD.dates, skip=2)

    
    ##################### PLOTTING APPROX LINEAGE ################
    
    ax_true_cases = Axis(g_seq_count[1:2,3])
    
    sim_freq = get_posterior(MS, "sim_freq", true)
    med, lQ, uQ = get_quants(sim_freq, ps)
    
    plot_data.sim_freqs = reduce(vcat, [m for m in eachcol(med)])
    barplot!(ax_true_cases, plot_data.date, plot_data.cases .* plot_data.sim_freqs, 
        stack = plot_data.lineage_num,
        color = color_vec)    
    
    add_monthly_dates!(ax_true_cases, LD.dates, skip=2)

    linkyaxes!(ax_obs_cases, ax_true_cases)
    hideydecorations!(ax_true_cases, grid = false)

    for (label, layout) in zip(["(a)", "(b)", "(c)"], [g_cases_rt, g_seq_count, g_variant])
    Label(layout[1, 1, TopLeft()], label,
        textsize = fontsize * 1.8,
        padding = (0, 5, 5, 0),
        font = font,
        halign = :right)
    end
    
    return fig
end

function figure_2(MS, LD, colors; figsize=(1800, 1800), fontsize=32, font="Helvetica")
    dates_num, seed_L, forecast_L, N_lineage = unpack_data(MS)
    dates = LD.dates 
    
    seq_labels = LD.seq_names
    #lineage_map = get_sequence_map(seq_labels)
    #WHO_seq_names = [lineage_to_WHO[lineage] for lineage in seq_labels]
    
    cases = MS.data["cases"]
    counts =  MS.data["num_sequenced"]
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = figsize, fontsize = fontsize, font = font)
    
    
    # Posterior smooth prevalence 
    g_smooth = fig[1:4,1] = GridLayout()
    ax_smooth = Axis(g_smooth[1,1], ylabel = "Posterior Smoothed Cases")
    plot_cases!(ax_smooth, MS)
    plot_smoothed_EC!(ax_smooth, MS)
    add_monthly_dates!(ax_smooth, dates)
    hidexdecorations!(ax_smooth, grid = false)

    # Frequency plot 
    g_freq = fig[5:8, 1] = GridLayout()
    ax_freq = Axis(g_freq[1,1],ylabel = "Posterior Lineage Frequency")
    plot_observed_frequencies!(ax_freq, MS; colors = colors, size = N -> 1.5 * sqrt(N))
    plot_lineage_frequency!(ax_freq, MS; colors = colors)
    add_monthly_dates!(ax_freq, dates)
    
    linkxaxes!(ax_smooth, ax_freq)
    
    # Posterior lineage cases
    g_smooth_lin = fig[1:4,2] = GridLayout()
    ax_smooth_lin = Axis(g_smooth_lin[1,1],ylabel = "Posterior Lineage Cases")
    
    plot_lineage_prev!(ax_smooth_lin, MS; colors = colors)
    add_monthly_dates!(ax_smooth_lin, dates)
    
    
    # Link y-axes with other smooth
    hideydecorations!(ax_smooth_lin, grid = false)
    linkyaxes!(ax_smooth, ax_smooth_lin)
    
    
    # Effective Reproductive Number Panel
    g_Rt = fig[5:6, 2] = GridLayout()
    ax_Rt = Axis(g_Rt[1,1], ylabel = L"R_t")
    
    plot_lineage_R_censored!(ax_Rt, MS, colors = colors)
    add_monthly_dates!(ax_Rt, dates)
    
    g_growth = fig[7:8, 2] = GridLayout()
    ax_growth = Axis(g_growth[1,1],ylabel = "Growth Advantage")
    
    plot_growth_advantage!(ax_growth, MS, colors = colors)
    ax_growth.xtickformat = xs -> seq_labels[convert.(Int, xs)]

    # Adding legend
    elements = [PolyElement(polycolor = colors[i]) for (i,s) in enumerate(seq_labels)]
    fig[9,1:2] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)", "(e)"], [g_smooth, g_smooth_lin, g_freq, g_Rt, g_growth])
        Label(layout[1, 1, TopLeft()], label,
            textsize = fontsize*1.8,
            padding = (0, 5, 5, 0),
            font = font,
            halign = :right)
    end
    
    return fig
end
