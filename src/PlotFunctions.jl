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
lineage_colors =["#2e5eaa", "#5adbff", "#b4c5e4","#f5bb00","#56e39f", "#9e4244", "#f03a47", "#808080"] 

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

function unpack_params(SoI, states_dict)
    seq_labels = vcat(states_dict[SoI]["seq_labels"]...)
    dates = states_dict[SoI]["date"]
    dates_num = collect(1:length(dates))
    seed_L = states_dict[SoI]["stan_data"]["seed_L"]
    #dates_num_ws = collect((-seed_L+1):(length(dates)))
    #dates_ws = dates[1] + Day.(dates_num_ws)
    return seq_labels, dates, dates_num, seed_L
end

# lineage to index map
function get_sequence_map(seq_labels)
    lineage_map = Dict()
    for (i, lineage) in enumerate(sort(seq_labels))
        lineage_map[lineage] = i
    end
    return lineage_map
end

# Interval-Methods
function plot_R(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    R = get_posterior(states_dict, SoI, "R.", true)
    lQ, uQ, med = sim_stats_multi(R, lQuants, uQuants)
    lQ, uQ, med = parse_by_deme(lQ, uQ, med)
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Effective Reproduction Number")

    lines!(ax, dates_num, fill(1., length(dates_num)), color = "black", linestyle=:dash)  

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[lineage][:, i], uQ[lineage][:,i], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[lineage][:,1], color = "black", label = "Median")
    end
    
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig 
end

function plot_sim_freq(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    sim_freq = get_posterior(states_dict, SoI, "sim_freq", true)
    lQ, uQ, med = sim_stats_multi(sim_freq, lQuants, uQuants)
    lQ, uQ, med = parse_by_deme(lQ, uQ, med)

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Lineage Frequencies")

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[lineage][:,i], uQ[lineage][:,i], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[lineage][:,1], color = "black", linewidth = 1.5, label = "Median")
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig 
end

function plot_post_pred_freq(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    sample_freq = states_dict[SoI]["stan_data"]["num_sequenced"] ./ sum(states_dict[SoI]["stan_data"]["num_sequenced"], dims = 2)

    obs_freq = get_posterior(states_dict, SoI, "obs_freqs", true)
    lQ, uQ, med = sim_stats_multi(obs_freq, lQuants, uQuants)
    lQ, uQ, med = parse_by_deme(lQ, uQ, med)

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior predictive sample frequencies")

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[lineage][:,i], uQ[lineage][:,i], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[lineage][:,1], color = "black", linewidth = 1.5, label = "Median")
        CairoMakie.scatter!(ax, dates_num,sample_freq[:,lineage],
            color = (this_color, 1.0),
            strokewidth = 0.5)
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig
end

function plot_growth_advantage(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    WHO_seq_names = [lineage_to_WHO[lineage] for lineage in seq_labels]
    
    v = get_posterior(states_dict, SoI, "v", false)
    v = vcat(hcat(v...)')
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Inferred Growth Advantage")

    hlines!(ax, [1.0], color = :black, linestyle = :dash)
    for (lineage, name) in enumerate(seq_labels[1:end-1])
        this_color = lineage_colors[lineage_map[name]]
        CairoMakie.violin!(ax, fill(lineage, size(v, 1)), exp.(v[:, lineage]), 
            color = this_color, 
            orientation=:vertical,
            width = 0.25,
            strokewidth = 1.5,
            strokecolor = :black)
        CairoMakie.scatter!(ax, fill(lineage, size(v, 1)), exp.(v[:, lineage]), color = (:black, 0.1))
    end

    ax.xticks = 1:length(seq_labels)
    ax.xtickformat = xs -> WHO_seq_names[convert.(Int,xs)]
    #CairoMakie.ylims!(ax, (0.9, 2.5))

    elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    fig[2,1] = Legend(fig, elements, WHO_seq_names, "", orientation = :horizontal, tellwidth = false, tellheight = true)
    fig
end

function plot_lineage_I_prev(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    scaled_prev = get_posterior(states_dict, SoI, "scaled_prev.", true)
    lQ, uQ, med = sim_stats_multi(scaled_prev, lQuants, uQuants)
    lQ, uQ, med = parse_by_deme(lQ, uQ, med)
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Smoothed Cases")

    barplot!(ax, dates_num, states_dict[SoI]["stan_data"]["cases"], color = (:black, 0.3))

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[lineage][:,i], uQ[lineage][:,i], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[lineage][:,1], color = "black", linewidth = 1.5, label = "Median")
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)
    
    fig
end

function plot_EC_smooth(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    EC_smooth = get_posterior(states_dict, SoI, "EC_smooth.", false)
    EC_smooth = vcat(hcat(EC_smooth...))
    med = median(EC_smooth, dims = 2)
    lQ = vcat([quantile(vi, lQuants) for vi in eachrow(EC_smooth)]...)
    uQ = vcat([quantile(vi, uQuants) for vi in eachrow(EC_smooth)]...) 

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Smoothed Cases")

    barplot!(ax, dates_num, states_dict[SoI]["stan_data"]["cases"], color = (:black, 0.3))
    # Plot credible intervals
    for i in reverse(1:length(ps))
        band!(ax, dates_num, 
            lQ[:,i],  uQ[:,i], 
            color = (:purple, alphas[i]), 
            label = "$(Int(ps[i] * 100))% CI")
    end

    # Add median
    lines!(ax, dates_num, med[:,1], color = "black", linewidth = 1.5, label = "Median")
    
    # TIME AXIS 
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
    fig
end

function plot_post_pred_seq_counts(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)

    sample_counts = states_dict[SoI]["stan_data"]["num_sequenced"]
    obs_counts = get_posterior(states_dict, SoI, "obs_counts", true)
    lQ, uQ, med = sim_stats_multi(obs_counts, lQuants, uQuants)
    lQ, uQ, med = parse_by_deme(lQ, uQ, med) 
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 1280), fontsize = 24)
    supertitle = fig[1, 1] = Label(fig, "Posterior predictive lineage counts",
        textsize = 24, color = :black, orientation = :horizontal, tellwidth = false, tellheight = true)
    ticks, _ = get_nice_ticks(dates)


    ax = []
    for i in 1:length(seq_labels)
        ax_now = Axis(fig)
        push!(ax, ax_now)
        fig[i+1, 1] = ax_now
        ax_now.xticks = ticks
        ax_now.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
    end

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        CairoMakie.barplot!(ax[lineage], dates_num,sample_counts[:,lineage],
            color = (:black, 0.2))

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax[lineage], dates_num, 
                lQ[lineage][:,i], uQ[lineage][:,i], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax[lineage], dates_num, med[lineage][:,1], color = "black", linewidth = 1.5, label = "Median")

      if lineage != length(seq_labels)
            hidexdecorations!(ax[lineage], grid = false) 
        end
    end

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[length(seq_labels)+2,1] = Legend(fig, elements, WHO_seq_names, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig
end

function plot_EC(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)

    EC = get_posterior(states_dict, SoI, "EC.", false)
    EC = vcat(hcat(EC...))
    med = median(EC, dims = 2)
    lQ = vcat([quantile(vi, lQuants) for vi in eachrow(EC)]...)
    uQ = vcat([quantile(vi, uQuants) for vi in eachrow(EC)]...) 

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Expected Cases")

    barplot!(ax, dates_num, states_dict[SoI]["stan_data"]["cases"], color = (:black, 0.3))
    # Plot credible intervals
    for i in reverse(1:length(ps))
        band!(ax, dates_num, 
            lQ[:,i],  uQ[:,i], 
            color = (:purple, alphas[i]), 
            label = "$(Int(ps[i] * 100))% CI")
    end

    # Add median
    lines!(ax, dates_num,med[:,1], color = "black", linewidth = 1.5, label = "Median")

    #
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    fig
end

# HDI-Methods
function plot_R_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    R = get_posterior(states_dict, SoI, "R.", true)
    med, lQ, uQ = get_quants(R, ps)
    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Effective Reproduction Number")

    lines!(ax, dates_num, fill(1., length(dates_num)), color = "black", linestyle=:dash)  

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals\
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:, lineage], uQ[i][:, lineage], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", label = "Median")
    end
    
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig 
end

function plot_sim_freq_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    sim_freq = get_posterior(states_dict, SoI, "sim_freq", true)
    med, lQ, uQ = get_quants(sim_freq, ps)

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Lineage Frequencies")

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[lineage][:,lineage], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig 
end

function plot_post_pred_freq_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    sample_freq = states_dict[SoI]["stan_data"]["num_sequenced"] ./ sum(states_dict[SoI]["stan_data"]["num_sequenced"], dims = 2)

    obs_freq = get_posterior(states_dict, SoI, "obs_freqs", true)
    med, lQ, uQ = get_quants(obs_freq, ps)

    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior predictive sample frequencies")

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[i][:,lineage], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
        CairoMakie.scatter!(ax, dates_num,sample_freq[:,lineage],
            color = (this_color, 1.0),
            strokewidth = 0.5)
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig
end

function plot_lineage_I_prev_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    scaled_prev = get_posterior(states_dict, SoI, "scaled_prev.", true)
    med, lQ, uQ = get_quants(scaled_prev, ps)

    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Smoothed Cases")

    barplot!(ax, dates_num, states_dict[SoI]["stan_data"]["cases"], color = (:black, 0.3))

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax, dates_num, 
                lQ[i][:,lineage], uQ[i][:,lineage], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax, dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")
    end

    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])

    elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[2,1] = Legend(fig, elements, seq_labels, "", orientation = :horizontal, tellwidth = false, tellheight = true)
    
    fig
end

function plot_EC_smooth_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)
    
    EC_smooth = get_posterior(states_dict, SoI, "EC_smooth.", false)
    med, lQ, uQ = get_quants(EC_smooth, ps)


    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 800), fontsize = 24)
    ax = fig[1, 1] = Axis(fig,ylabel = "Posterior Smoothed Cases")

    barplot!(ax, dates_num, states_dict[SoI]["stan_data"]["cases"], color = (:black, 0.3))
    # Plot credible intervals
    for i in reverse(1:length(ps))
        band!(ax, dates_num, 
            lQ[i],  uQ[i], 
            color = (:purple, alphas[i]), 
            label = "$(Int(ps[i] * 100))% CI")
    end

    # Add median
    lines!(ax, dates_num, med, color = "black", linewidth = 1.5, label = "Median")
    
    # TIME AXIS 
    ticks, _ = get_nice_ticks(dates)
    ax.xticks = ticks
    ax.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
    fig
end

function plot_post_pred_seq_counts_HDI(SoI, states_dict)
    seq_labels, dates, dates_num, seed_L = unpack_params(SoI, states_dict)
    lineage_map = get_sequence_map(seq_labels)

    sample_counts = states_dict[SoI]["stan_data"]["num_sequenced"]
    obs_counts = get_posterior(states_dict, SoI, "obs_counts", true)
    med, lQ, uQ = get_quants(obs_counts, ps)

    
    fig = Figure(backgroundcolor = RGBf0(1., 1., 1.), resolution = (1280, 1280), fontsize = 24)
    supertitle = fig[1, 1] = Label(fig, "Posterior predictive lineage counts",
        textsize = 24, color = :black, orientation = :horizontal, tellwidth = false, tellheight = true)
    ticks, _ = get_nice_ticks(dates)


    ax = []
    for i in 1:length(seq_labels)
        ax_now = Axis(fig)
        push!(ax, ax_now)
        fig[i+1, 1] = ax_now
        ax_now.xticks = ticks
        ax_now.xtickformat = xs -> Dates.monthabbr.(dates[convert.(Int, xs)])
    end

    for (lineage, name) in enumerate(seq_labels)
        this_color = lineage_colors[lineage_map[name]]

        CairoMakie.barplot!(ax[lineage], dates_num,sample_counts[:,lineage],
            color = (:black, 0.2))

        # Plot credible intervals
        for i in reverse(1:length(ps))
            band!(ax[lineage], dates_num, 
                lQ[i][:,lineage], uQ[lineage][:,lineage], 
                color = (this_color, alphas[i]), 
                label = "$(Int(ps[i] * 100))% CI")
        end

        # Add median
        lines!(ax[lineage], dates_num, med[:,lineage], color = "black", linewidth = 1.5, label = "Median")

      if lineage != length(seq_labels)
            hidexdecorations!(ax[lineage], grid = false) 
        end
    end

    #elements = [PolyElement(polycolor = lineage_colors[lineage_map[l]]) for l in seq_labels]
    #fig[length(seq_labels)+2,1] = Legend(fig, elements, WHO_seq_names, "", orientation = :horizontal, tellwidth = false, tellheight = true)

    fig
end
