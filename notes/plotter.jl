### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0a46b1e2-b862-495d-a069-76ac6b27b556
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(
        m,
        Expr(
            :toplevel,
            :(eval(x) = $(Expr(:core, :eval))($name, x)),
            :(include(x) = $(Expr(:top, :include))($name, x)),
            :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
            :(include($path)),
        ),
    )
    m
end;

# ╔═╡ f5dbf536-247b-11ed-1663-2134cf17d1fb
begin
    using Pkg
    Pkg.activate()

    using LinearAlgebra: diag
    using KadanoffBaym, LangrethsRules

    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Plots, LaTeXStrings

    using Serialization

    using PlutoUI
end;

# ╔═╡ 95e5d72e-aaa8-49fa-a473-d96507f1ba7e
begin
    # Plots.pyplot(); Plots.PyPlot.plt.style.use(["science"])
    Plots.gr()

    # Plots.default()
    Plots.default(
        fontfamily = "Computer Modern",
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        tick_direction = :in,
        minorticks = true,
        framestyle = :box,
        size = (600, 450),
        titlefontsize = 18,
        legendfontsize = 12,
        tickfontsize = 12,
        grid = true,
        #     dpi=100,
    )

    my_colors = Dict(
        :c => Plots.Colors.colorant"#0C5DA5",
        :dω => Plots.Colors.colorant"#FF9500",
        :dτ => Plots.Colors.colorant"#00B945",
        :ωK => Plots.Colors.colorant"#9e9e9e",
        :εd => Plots.Colors.colorant"#845B97",
        :ω0 => Plots.Colors.colorant"#FF2C00",
    )
end;

# ╔═╡ fd5e901c-ebf8-493e-86ef-5a0f3ac1a73b
begin
    (; Ts, ωs, data, params) = deserialize("../data/test0.jls")
    (; ρd_ω, GdL_ω, ρc_ω, GcL_ω, ργ0_ω, GγL0_ω, ργ_ω, GγL_ω, GγL_vac) = data
end;

# ╔═╡ 1efa1af7-bd2d-49c2-8f4a-4cb65fc63255
begin
	# t0 index
	i_t0 = findmin(abs, Ts)[2]

	∫dω(gf) = sum(gf, dims=1) * (ωs[2] - ωs[1]) / (2π) |> x -> dropdims(x, dims=1)
	
	# Calculate particle number
	nγ0 = -imag(∫dω(GγL0_ω))
	nγ = -imag(∫dω(GγL_ω))
	nγ .-= nγ[end]

	nγ_vac = -imag(∫dω(GγL_vac))

	# Kondo points
	w = 20
	i1 = argmin(abs.(ωs .+ w * params.ωK))
	i2 = argmin(abs.(ωs .- w * params.ωK))

	# Integral over Kondo weight
	∫dωK(gf) = sum(gf[i,:] for i in i1:i2) * (ωs[2] - ωs[1]) / (2π)
end

# ╔═╡ 330f7766-daf8-4d47-a38a-b7797b5e6264
let
    cecu6_half_bandwidth = 2.0 # eV
    cdos_half_bandwidth = 2.0
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth

    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 3)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)

    md"""
    #### Summary
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the  parameters are
    1. ``\varepsilon_d =`` $(ω_to_eV(params.ε0f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(params.V0)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(params.ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(params.ωK)) ``ps`` (Kondo temperature & timescale)
    4. ``T = `` $(ω_to_K(inv(params.β)))  ``K`` (system temperature)
    and the gaussian-shaped pulse parameters were
    5. ``\omega = `` $(ω_to_THz(params.ω0)) ``THz`` (pulse energy)
    6. ``\Omega = `` $(ω_to_THz(params.Ω0)) ``THz``  (pulse bandwidth: $(ω_to_ps(params.Ω0/6)) ``ps``)
    7. ``\Gamma = `` $(ω_to_eV(params.Γ0)) ``eV`` (light-matter coupling)
    8. ``N_\gamma = `` $(round(params.Nγ, digits=3)) ``n_\gamma`` (number of photons)

    The system will be integrated in the interval (0, $(ω_to_ps(inv(Ts[end])))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(params.tγ))) ``ps``.
    """
end

# ╔═╡ c5b0d68b-0b1b-4795-ac59-521988dff946
@bind i_max_pulse Slider(eachindex(Ts))

# ╔═╡ fde3dc61-74e7-4123-a747-beb6972888dc
let
    Plots.gr()

    plot(
        Ts,
        nγ0,
        label = L"n_\gamma^\mathrm{in}",
        xlabel = L"T\;D",
        line = (my_colors[:c], 4.0, 0.5),
        legend = :topright,
        left_margin = 5Plots.mm,
        right_margin = 15Plots.mm,
        title = "Photon output",
		ylims=(-0.001, 0.05),
		# xlims = (-20, 20.),
		xlims = (Ts[1], Ts[end]),
    )

    plot!(
        Ts,
        nγ;
        color = my_colors[:ω0],
        # label="out",
        label = L"n_\gamma^\mathrm{out}",
    )

	vspan!(
        params.tγ .+ params.τK .* [-0.5, +0.5];
        label = L"\tau_K",
        fillcolor = :grey,
        alpha = 0.1,
        linewidth = 0,
        markersize = 0,
    )

	vline!([Ts[i_max_pulse]]; label = nothing, line = (:dot, 1.5, :grey, 0.75))
	
	# plot!(twinx(),
	# 	Ts,
	# 	nγ_vac;
	# )
	
	# plot!(twinx(),
	# 	Ts,
	# 	-imag(∫dωK(ρd_ω));
	# 	label = nothing,
	# 	linecolor = my_colors[:dτ],
	# 	xlims=(Ts[1], Ts[end]),
	# )

	# plot!(twinx(),
	# 	Ts,
	# 	imag(∫dωK(GdL_ω));
	# 	label = nothing,
	# 	linecolor = :grey,
	# 	xlims=(Ts[1], Ts[end]),
	# )
end

# ╔═╡ 245a0b54-860d-411a-bca8-9b5e3cc69dc7
# Comparison of frequency *projected* solution with time solution
let
    Plots.gr()

    plot(
        ωs,
        -imag(ρd_ω[:, i_t0]);
        label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
        xlabel = L"\omega / D",
        xlims = (-1.0, 1.0),
        ylims = (0.0, Inf),
        legend = :topright,
		legendfontsize = 10,
        linecolor = my_colors[:dω],
        title = "Kondo peak melt",
    )

    plot!(
        ωs,
        imag(GdL_ω[:, i_t0]);
        label = nothing,
        fill = (0, my_colors[:dω], 0.5),
        linecolor = nothing,
    )

    plot!(
        ωs,
        -imag(ρd_ω[:, i_max_pulse]);
        label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
    )

    plot!(
        ωs,
        imag(GdL_ω[:, i_max_pulse]),
        label = nothing,
        fill = (0, my_colors[:dτ], 0.5),
        linecolor = nothing,
    )

    vline!([params.ε0f]; label = L"\varepsilon_d", line = (:dot, 1.5, my_colors[:εd], 0.75))

    vline!(
        [-params.ω0, params.ω0];
        label = L"\omega_0",
        line = (:dot, 1.5, my_colors[:ω0], 0.75),
    )

	p1 = plot!(
        ωs,
        (-GdL_ω[:, i_t0] ./ ρd_ω[:, i_t0]) |> real;
        label = L"f_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
		linecolor = my_colors[:dω],
        title = "Distribution functions",
		xlabel = L"\omega / D",
		legend=:bottomleft,
		xlims = (-0.2, 0.2),
		ylims = (-0.1, 1.1),
		inset = (1, bbox(0.63, 0.53, 0.35, 0.35)),
		subplot = 2,
		titlefontsize = 10,
		legendfontsize = 7,
		xtickfont = font(5),
		ytickfont = font(5),
		xguidefontsize = 5,
    )

	plot!(p1,
        ωs,
        (-GdL_ω[:, i_max_pulse] ./ ρd_ω[:, i_max_pulse]) |> real;
        label = L"f_d(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
		# inset = (1, bbox(0.6, 0.5, 0.3, 0.4)),
		subplot = 2,
    )
	
    # vspan!(params.ωK * [-1 / 2, 1 / 2]; 
    #     fillcolor = my_colors[:ωK], 
    #     alpha = 0.3, 
    #     label = L"\omega_K", 
    #     linewidth = 0, 
    #     markersize = 0
    # )
end

# ╔═╡ 54cc7940-6a69-46d1-93ae-bb705e0ba17d
# Comparison of frequency *projected* solution with time solution
let
    Plots.gr()

    plot(
        ωs,
        -imag(ργ0_ω[:, i_max_pulse]);
        label = L"-2\,\operatorname{Im}\,G^R_\gamma(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        xlabel = L"\omega / D",
        xlims = (0.0, 0.2),
        ylims = (0.0, Inf),
        legend = :topright,
		legendfontsize = 10,
        linecolor = my_colors[:dω],
        title = "Photonic spectral functions",
    )

    plot!(
        ωs,
        -imag(GγL0_ω[:, i_max_pulse]);
        label = nothing,
        fill = (0, my_colors[:dω], 0.5),
        linecolor = nothing,
    )

    plot!(
        ωs,
        -imag(ργ_ω[:, i_max_pulse]);
        label = L"-2\,\operatorname{Im}\,G^R_\gamma(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
    )

    plot!(
        ωs,
        -imag(GγL_ω[:, i_max_pulse]),
        label = nothing,
        fill = (0, my_colors[:dτ], 0.5),
        linecolor = nothing,
    )

    # vline!([params.ε0f]; label = L"\varepsilon_d", line = (:dot, 1.5, my_colors[:εd], 0.75))

    vline!(
        [-params.ω0, params.ω0];
        label = L"\omega_0",
        line = (:dot, 1.5, my_colors[:ω0], 0.75),
    )

	p1 = plot!(
        ωs,
        (GγL_ω[:, i_t0] ./ ργ_ω[:, i_t0]) |> real;
        label = L"f_\gamma(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
		linecolor = my_colors[:dω],
        title = "Distribution functions",
		xlabel = L"\omega / D",
		legend=:bottomleft,
		xlims = (-0.2, 0.2),
		ylims = (-1.1, 1.1),
		inset = (1, bbox(0.63, 0.53, 0.35, 0.35)),
		subplot = 2,
		titlefontsize = 10,
		legendfontsize = 7,
		xtickfont = font(5),
		ytickfont = font(5),
		xguidefontsize = 5,
    )

	plot!(p1,
        ωs,
        (GγL_ω[:, i_max_pulse] ./ ργ_ω[:, i_max_pulse]) |> real;
        label = L"f_\gamma(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
		# inset = (1, bbox(0.6, 0.5, 0.3, 0.4)),
		subplot = 2,
    )
	
    # vspan!(params.ωK * [-1 / 2, 1 / 2]; 
    #     fillcolor = my_colors[:ωK], 
    #     alpha = 0.3, 
    #     label = L"\omega_K", 
    #     linewidth = 0, 
    #     markersize = 0
    # )
end

# ╔═╡ 5b4ddb44-99fd-4db4-9008-d6215154a152
let
    plot(
        ωs,
        (-GdL_ω[:, i_t0] ./ ρd_ω[:, i_t0]) |> real;
        label = L"f_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
		linecolor = my_colors[:dω],
        title = "Distribution functions",
    )

    plot!(
        ωs,
        (-GdL_ω[:, i_max_pulse] ./ ρd_ω[:, i_max_pulse]) |> real;
        label = L"f_d(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
    )

	plot!(
        ωs,
        (-GcL_ω[:, i_max_pulse] ./ ρc_ω[:, i_max_pulse]) |> real;
        label = L"f_c(\omega, T=%$(round(Ts[i_max_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:c],
    )

    # plot!(
    #     ωs,
    #     1.0 ./ (exp.(params.β .* ωs) .+ 1);
    #     label = L"f_{eq}(\omega)",
    #     xlabel = L"\omega / D",
    #     xlims = (ωs[1], ωs[end]),
    #     legend = :topright,
    #     line = (:dot, :grey),
    # )

    ylims!(-0.1, 1.1)
    xlims!(-0.3, 0.3)
end

# ╔═╡ 4226c527-926f-4a1f-b9a3-91891e1ea95a
# ╠═╡ disabled = true
#=╠═╡
let
    Plots.gr()

    Gd_τ = let
		τs, ρd_τ = EchoPulse.ft(ωs, ρd_ω[:, i_t0]; inverse = true)
        plot(
            τs,
            -imag(ρd_τ);
            label = L"-2\,\operatorname{Im}\,G^R_d(\tau, T=%$(round(Ts[i_t0])))_{\textrm{W}}",
            title = "Spectral functions",
            xlabel = L"\tau \;D",
            xlims = params.τK .* (-2 / 3, 2 / 3),
            legend = :topright,
            alpha = 1.0,
            ms = 1.0,
            linecolor = my_colors[:dω],
            left_margin = 5Plots.mm,
            right_margin = 15Plots.mm,
        )
    end

    Gd_τ = let
		τs, ρd_τ = EchoPulse.ft(ωs, ρd_ω[:, i_max_pulse]; inverse = true)
		plot!(
	        τs,
	        -imag(ρd_τ);
	        label = L"-2\,\operatorname{Im}\,G^R_d(\tau, T=%$(round(Ts[i_max_pulse])))_{\textrm{W}}",
	        ms = 2.0,
	        linecolor = my_colors[:dτ],
	    )
	end

    Gd_τ_kondo = let
        # indices of the "Kondo peak"
        w = 20
        i1 = argmin(abs.(ωs .+ w * params.ωK))
        i2 = argmin(abs.(ωs .- w * params.ωK))

        # Fourier transform to τ-space just the peak
        kondo_ft = EchoPulse.ft(ωs[i1:i2], -ρd_ω[i1:i2, i_t0]; inverse = true)

        Gd_τ_kondo = plot(
            kondo_ft[1],
            imag(kondo_ft[2]);
            label = L"-2\,\operatorname{Im}\,G^{R}_d(\tau)",
            xlabel = L"\tau \;D",
            linecolor = my_colors[:dω],
            legend = :topright,
            xlims = params.τK .* (-2 / 3, 2 / 3),
            title = "Kondo peak duration",
        )

    #     # hline!([maximum(imag(kondo_ft[2])) * exp(-1)], label=nothing)

        plot!(
            Gd_τ_kondo,
            [],
            [];
            label = L"-2\,\operatorname{Im}\,G^{0,R}_\gamma(\tau)",
            linecolor = my_colors[:ω0],
        )

		τs, ργ0_τ = EchoPulse.ft(ωs, ργ0_ω[:, i_max_pulse]; inverse = true)
        plot!(
            twinx(),
            τs,
            -imag(ργ0_τ);
            linecolor = my_colors[:ω0],
            xaxis = nothing,
            legend = nothing,
            xlims = params.τK .* (-2 / 3, 2 / 3),
        )

        scatter!(
            ωs[i1:i2],
            -imag(ρd_ω[i1:i2, i_t0]);
            xlabel = L"\omega / D",
            markersize = 2.0,
            marker = (1.0, :grey),
            xtickfont = font(5),
            ytickfont = font(5),
            xguidefontsize = 5,
            label = nothing,
            inset = (1, bbox(0.05, 0.1, 0.3, 0.4)),
            subplot = 3,
        )
    end

    vspan!(
        Gd_τ_kondo,
        params.τK * [-1 / 2, 1 / 2];
        fillcolor = my_colors[:ωK],
        alpha = 0.3,
        label = L"\tau_K",
        linewidth = 0,
        markersize = 0,
    )

    plot(Gd_τ, Gd_τ_kondo; layout = grid(2, 1))
    # savefig("equilibrium-comparison.pdf")
end
  ╠═╡ =#

# ╔═╡ 3da081a7-0c7a-4bed-8fe7-246f8b4ca710
# ╠═╡ disabled = true
#=╠═╡
let
    Plots.gr()

    # Spectral function
    h1 = heatmap(
        Ts,
        ωs,
        -imag(ρd_ω);
        title = L"-2\,\operatorname{Im}\,G^R_d(\omega,T)",
        xlabel = L"T\;D",
        ylabel = L"\omega /D",
        minorticks = true,
        tick_direction = :out,
        ylims = (-min(2.0, abs(minimum(ωs))), min(2.0, maximum(ωs))),
        # legend = :none,
        c = :magma,
        # rightmargin = 5Plots.mm,
    )
end
  ╠═╡ =#

# ╔═╡ 3013df1c-cbf6-4607-b6d1-0b0a571b345d
# ╠═╡ disabled = true
#=╠═╡
let
    # nγ0 = -imag(diag(Gγ0.L))
    # nγ = -imag(diag(Gγ.L))
    # nγ .-= nγ[i_t0] # subtract "background" emissions

    ylims_ργ = (0.0, 0.05)#extrema(nγ0)
    ylims_ρd = extrema(-imag(ρd_ω[:, i_t0]))

    function plotter(i_t)
        h1 = plot(
            ωs,
            -imag(ρd_ω[:, i_t0]);
            label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
            xlabel = L"\omega / D",
            xlims = (-1.0, 1.0),
            legend = :topright,
            linecolor = my_colors[:dω],
        )

        h1 = plot!(
            ωs,
            imag(GdL_ω[:, i_t0]);
            label = nothing,
            fill = (0, my_colors[:dω], 0.5),
            linecolor = nothing,
        )

        h1 = plot!(
            ωs,
            -imag(ρd_ω[:, i_t]);
            label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t])))_{\tilde{\textrm{W}}}",
            linecolor = my_colors[:dτ],
            ylims = ylims_ρd,
            margin = 0Plots.mm,
        )

        h1 = plot!(
            ωs,
            imag(GdL_ω[:, i_t]),
            label = nothing,
            fill = (0, my_colors[:dτ], 0.5),
            linecolor = nothing,
        )

        # sa(t) = [
        #     range(0, 1; length = max(t - 5, 2))
        #     ones(max(1, min(5, t)))
        #     zeros(length(ts) - t)
        # ]
        # sa(t) = range(0, 1; length=t)
        # sa(t) = let
        # 	v = max(1, min(5,t))
        # 	[
        # 		t-v == 1 ? [0.0] : range(0, 1; length=max(t-v, 0))
        # 		ones(v)
        # 	]
        # end
        sa(t) =
            let
                t = t - i_t0 + 1
                v = max(1, min(5, t))
                [
                    1 # needed to get the correct label marker
                    t - v == 1 ? [0.0] : range(0, 1; length = max(t - v, 0))
                    ones(v - 1)
                ]
            end

        # Occupations
        p1 = plot(
            Ts[i_t0:i_t0],
            nγ0[i_t0:i_t0];
            label = L"n^\mathrm{in}_\gamma",
            line = (my_colors[:c], 2.5),
            grid = false,
            seriesalpha = 0.8sa(i_t),
            ylims = ylims_ργ,
            xlims = (Ts[i_t0], Ts[end]),
            legend = :topright,
        )

        p1 = plot!(
            Ts[i_t0:i_t0],
            nγ[i_t0:i_t0];
            line = (my_colors[:ω0], 2.5),
            label = L"n_\gamma^\mathrm{out}",
        )

        # Kondo time-scale
        p1 = vspan!(
            p1,
            params.tγ .+ params.τK .* [-0.5, +0.5];
            fillcolor = :grey,
            alpha = 0.1,
            linewidth = 0,
            markersize = 0,
            label = L"\tau_K",
        )

        plt = twinx()

        p1 = plot!(
            plt,
            Ts[i_t0:i_t],
            nγ0[i_t0:i_t];
            label = nothing,
            line = (my_colors[:c], 4.0, 0.5),
            legend = :topright,
            xlabel = L"T\;D",
            xlims = (Ts[i_t0], Ts[end]),
            ylims = ylims_ργ,
            grid = false,
            seriesalpha = 0.8sa(i_t),
            margin = 5Plots.mm,
        )

        plot!(
            Ts[i_t0:i_t],
            nγ[i_t0:i_t];
            label = nothing,
            line = (my_colors[:ω0], 2.0, 0.8),
            seriesalpha = 0.8sa(i_t),
        )

        return plot(h1, p1; size = 1 // 3 .* (1920, 1080), layout = @layout grid(2, 1))
    end

    # anim = @animate for t in range(i_t0, length(ts), step = 4)
    #     iszero(mod(t, 10)) ? (@info ts[t]) : nothing
    #     plotter(t)
    # end

    # gif(anim, "anim_fps15.mp4", fps = 24)
end;
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─0a46b1e2-b862-495d-a069-76ac6b27b556
# ╟─f5dbf536-247b-11ed-1663-2134cf17d1fb
# ╟─95e5d72e-aaa8-49fa-a473-d96507f1ba7e
# ╠═fd5e901c-ebf8-493e-86ef-5a0f3ac1a73b
# ╟─1efa1af7-bd2d-49c2-8f4a-4cb65fc63255
# ╟─330f7766-daf8-4d47-a38a-b7797b5e6264
# ╠═c5b0d68b-0b1b-4795-ac59-521988dff946
# ╟─fde3dc61-74e7-4123-a747-beb6972888dc
# ╟─245a0b54-860d-411a-bca8-9b5e3cc69dc7
# ╟─54cc7940-6a69-46d1-93ae-bb705e0ba17d
# ╟─5b4ddb44-99fd-4db4-9008-d6215154a152
# ╟─4226c527-926f-4a1f-b9a3-91891e1ea95a
# ╟─3da081a7-0c7a-4bed-8fe7-246f8b4ca710
# ╟─3013df1c-cbf6-4607-b6d1-0b0a571b345d
