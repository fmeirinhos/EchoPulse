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

    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Serialization

    using PlutoUI
	using Plots, LaTeXStrings
end;

# ╔═╡ 95e5d72e-aaa8-49fa-a473-d96507f1ba7e
begin
    Plots.gr()
    Plots.default(
        fontfamily = "Computer Modern",
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        tick_direction = :in,
        minorticks = true,
        framestyle = :box,
		# size = 1.5 .* (600, 450),
        size = 3 * 288.00 .* ((1 + sqrt(5))/2, 1 + 0.35),
        titlefontsize = 18,
		guidefontsize = 12,
		legendfontsize = 10,
		tickfontsize = 10,
        # grid = true,
		grid = false,
        # margin = 5Plots.mm
    )
	Plots.scalefontsizes(2.0)

    my_colors = Dict(
        :c => Plots.Colors.colorant"#0C5DA5",
        :dω => Plots.Colors.colorant"#FF9500",
        :dτ => Plots.Colors.colorant"#00B945",
        :ωK => Plots.Colors.colorant"#9e9e9e",
        :εd => Plots.Colors.colorant"#845B97",
        :ω0 => Plots.Colors.colorant"#FF2C00",
    )
end; 

# ╔═╡ 0190be17-7d73-4522-8cc3-d1500801c408
begin
	str = "run-run-run"
	lbl = "ω0, Ω0"
	list_j = [5.0e-3, 1.0e-3, 5.0e-2, 1e-1, 5.0e-1, 1e-4, 5e-4] # ω0
	list_k = [6.0e-2, 9.0e-2, 2e-1] # Ω0

	# str = "run-run"
	# lbl = "Ω0, ε0f"
	# list_j = [-0.35, -0.39]
	# list_k = [2e-1, 9.0e-2, 6e-2]

	# str = "rrun"
	# lbl = "nₐ, ε0f"
	# list_j = [-0.35, -0.39]
	# list_k = [1.0, 3.0, 5.0, 7.0, 10.]

	# str = "run-run-run-run"
	# lbl = "α, ε0f"
	# list_j = [-0.35, -0.39]
	# list_k = [sqrt(0.02),  sqrt(0.03), sqrt(0.04), sqrt(0.05)]

	# str = "new-run"
	# lbl = "β"
	# list_j = [Inf]
	# list_k = [1e0, 3e0, 7e0, 1e1, 1.5e1, 3e1, 5e1, 8.5e1, 1e2, 1.5e2, 1.9e2]
	
	d = Dict(enumerate([(x, y) for x in list_j for y in list_k]))
	d = sort(collect(d), by = x -> (x[2], x[1]))
	
	md"$(lbl) $(@bind r Select(collect(d)))"
end

# ╔═╡ 1efa1af7-bd2d-49c2-8f4a-4cb65fc63255
begin
	(; Ts, ωs, data, params) = deserialize("../data/$(str)-$(r-1).jls")
	(; ρd_ω, GdL_ω, ρc_ω, GcL_ω, ργ_ω, GγL_ω, GγL_vac, ΣcR_ω) = data
	
	# t0 index
	i_t0 = findmin(abs, Ts)[2]

	# Integral over all frequencies
	∫dω(gf) = sum(gf, dims=1) * (ωs[2] - ωs[1]) / (2π) |> x -> dropdims(x, dims=1)

	# Kondo points
	w = 3
	i1 = argmin(abs.(ωs .+ w * params.ωK))
	i2 = argmin(abs.(ωs .- w * params.ωK))
	
	# Integral over Kondo frequencies
	∫dωK(gf) = sum(gf[i,:] for i in i1:i2) * (ωs[2] - ωs[1]) / (2π)
	
	# Calculate photon number
	nγ0 = -imag([EchoPulse.GγL0_ext(t,t; model=params) for t in Ts])
	
	nγ = -imag(∫dω(GγL_ω))
	nγ .-= nγ[end]

	kondo_average_number = map(x -> ∫dωK(x)[1], eachcol(imag(GdL_ω)))
	kondo_maximum_number = map(x -> maximum(x[i1:i2]), eachcol(imag(GdL_ω)))
	kondo_average_spectral = map(x -> ∫dωK(-x)[1], eachcol(imag(ρd_ω)))
	kondo_maximum_spectral = map(x -> maximum(-x[i1:i2]), eachcol(imag(ρd_ω)))
end;

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
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the electronic parameters are
    1. ``\varepsilon_0^f =`` $(ω_to_eV(params.ε0f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(params.V0)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(params.ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(params.ωK)) ``ps`` (Kondo temperature & timescale)
    and the bath parameters were
    4. ``T = `` $(ω_to_K(inv(params.β)))  ``K`` (system temperature)
    5. ``\alpha_1 = `` $(ω_to_eV(params.α1)) ``eV`` (fermionic bath strength)
    6. ``\alpha_2 = `` $(ω_to_eV(params.α2 * params.Γ0)) ``eV`` (bosonic bath strength)
    7. ``\omega_\mathrm{cutoff} = `` $(ω_to_THz(params.cutoff)) ``THz`` (soft cutoff frequency of bosonic modes)
    and the gaussian pulse parameters were
	
    8. ``\omega_0 = `` $(ω_to_THz(params.ω0)) ``THz`` (pulse energy)
    9. ``\Omega_0 = `` $(ω_to_THz(params.Ω0)) ``THz``  (pulse bandwidth: $(ω_to_ps(params.Ω0)) ``ps``)
    10. ``\Gamma_0 = `` $(ω_to_eV(params.Γ0)) ``eV`` (light-matter coupling)
    11. ``N_\gamma = `` $(round(params.Nγ, digits=3)) (number of photons)
	
    The system will be integrated in the interval (0, $(ω_to_ps(inv(Ts[end])))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(params.tγ))) ``ps``.
    """
end

# ╔═╡ c5b0d68b-0b1b-4795-ac59-521988dff946
md"Current time: $(@bind i_pulse Slider(eachindex(Ts)))"

# ╔═╡ cec8ba06-98c8-48a8-a360-1b19cda1e9de
let
    Plots.gr()

    plot(
        Ts,
        nγ0,
        label = L"n_\gamma^\mathrm{in}",
        xlabel = L"t/ \tau_K",
        line = nothing,
		fill = (0, :grey, 0.5),
        legend = :topright,
        left_margin = 5Plots.mm,
        right_margin = 15Plots.mm,
        title = "Photon output",
		ylims=(0.0, Inf),
		xtick= params.tγ .- (-30:1:30) .* params.τK,
		xformatter = x -> (x - params.tγ) / params.τK,
		# xlims = (-20, 20.),
		xlims = (0.0, 800),
    )

    plot!(
        Ts,
        nγ;
        color = my_colors[:ω0],
		xlims = (0.0, Ts[end]),
		ylims= (-0.00, 0.03),
        # label="out",
        label = L"n_\gamma^\mathrm{out}",
    )

	# vspan!(
 #        params.tγ .+ params.τK .* [-0.5, +0.5];
 #        label = L"\tau_K",
 #        fillcolor = :grey,
 #        alpha = 0.1,
 #        linewidth = 0,
 #        markersize = 0,
 #    )

	vline!([Ts[i_pulse]]; label = nothing, line = (:dot, 1.5, :grey, 0.75))
end

# ╔═╡ e375a68f-a9d1-4104-8548-0b3858cc9d32
# Comparison of frequency *projected* solution with time solution
let
    Plots.gr()
	# Plots.pgfplotsx()
	
    plot(
        ωs,
        -imag(ρd_ω[:, i_t0]);
        label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
        xlabel = L"\omega / t",
        xlims = 3 .* (-1/3, 1/3),
        ylims = (0.0, Inf),
        legend = :topleft,
		legendfontsize = 10,
        linecolor = my_colors[:dω],
        # title = "Kondo peak melt",
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
        -imag(ρd_ω[:, i_pulse]);
        label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_pulse])))_{\tilde{\textrm{W}}}",
        linecolor = my_colors[:dτ],
    )

    plot!(
        ωs,
        imag(GdL_ω[:, i_pulse]),
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
        # title = "Distribution functions",
		# titlelocation = :center,
		xlabel = L"\omega / t",
		legend = :bottomleft,
		xlims = (-0.2, 0.2),
		ylims = (-0.1, 1.1),
		inset = (1, bbox(0.63, 0.53, 0.35, 0.35)),
		subplot = 2,
		titlefontsize = 10,
		legendfontsize = 7,
		xtickfontsize = 5,
		ytickfontsize = 5,
		xguidefontsize = 5,
    )

	p1 = plot!(p1,
        ωs,
        (-GdL_ω[:, i_pulse] ./ ρd_ω[:, i_pulse]) |> real;
        label = L"f_d(\omega, T=%$(round(Ts[i_pulse])))_{\tilde{\textrm{W}}}",
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
	# savefig(fig, "myfig.svg")
end

# ╔═╡ 712bac76-feba-44f7-a094-093e7f45a20f
# begin	
# 	is = [i_t0 + 200, 3250, 3350, 3450, 3550, 3700, 4600, 5500]
# 	ps = [plot_gf(i) for i in is]

# 	pp = plot(
#         Ts,
#         nγ0,
# 		line = nothing,
# 		fill = (0, :grey, 0.7),
#         label = L"\bar{n}_a",
#         legend = :topright,
# 		ylims = (0.0, 10.0),
# 		xlims = (0.0, 800),
#     )

# 	for i in is
# 		pp = vline!([Ts[i]]; 
# 			label = nothing, 
# 			line = (:dash, 2.0, :grey, 0.75)
# 		)
# 	end

# 	pp = vspan!(pp,
#         [600, 800];
#         label = nothing,
#         fillcolor = :red,
#         alpha = 0.1,
#         linewidth = 0,
#         markersize = 0,
# 		fillstyle = :/,
# 		linecolor = :black
#     )

# 	plot(ps..., pp, margin = 3Plots.mm, layout = @layout [grid(3, 3)])
# 	# savefig("gd.svg")
# end

# ╔═╡ 63e379c3-bfab-4e88-ba07-c58b6ff7fff6
function plot_gf(i; right_margin, bottom_margin, y_mirror, xaxis, label, y_axis)

	colourmap = cgrad(:viridis)

	my_colors[:dω] = colourmap[0.65]
	my_colors[:dτ] = colourmap[0.1]
	
	scale = 2

	plot(;
		inset = (1, bbox(0.67, 0.20, 0.30, 0.45)),
		right_margin,
		bottom_margin,
		# xaxis = nothing,
	)

	plot!([Inf], [Inf], line = (:white), label = L"\ ")
	
	plot!([Inf], [Inf],
		line = (:black, 1.0, 0.0),
		label = L"t/\tau_K",
		legend = :topleft,
        xlims = 3 .* (-1/3 + 0.001, 1/3 - 0.001),
        ylims = (0.0, 9.0),		# (yticks ? tuple() : (; yticks = nothing, ))...,
		# (xticks ? tuple() : (; xticks = nothing, ))...,
		xticks = range(-0.8, 0.8, length=5),
		yticks = y_axis ? range(0, 8, length=5) : nothing,
		# xticks = nothing,
		# (xaxis ? (; xaxis = nothing) : xaxis)...,
		# yaxis = nothing,
		xlabel = L"\omega / D",
		xformatter = x -> x / scale,
		y_mirror = y_mirror,
		# tickfontsize = 14
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
        imag(GdL_ω[:, i]),
        label = nothing,
        fill = (0, my_colors[:dτ], 0.5),
        linecolor = nothing,
    )

	plot!(
        ωs,
        -imag(ρd_ω[:, i_t0]);
		label = L"-\infty",
        line = (4.0, my_colors[:dω]),
    )

	plot!(
        ωs,
        -imag(ρd_ω[:, i]);
		label = L"%$(round((Ts[i] - params.tγ) / params.τK, digits=2))",
        line = (4.0, my_colors[:dτ]),
    )
	
    vline!([params.ε0f]; 
		# label = L"\varepsilon^f_0",
		label = nothing,
		line = (:dash, 2.0, :black, 0.2)
	)

	annotate!(-0.9, 8.1, (L"( %$label )", 24), :black)

	plot!(
        ωs,
        (-GdL_ω[:, i_t0] ./ ρd_ω[:, i_t0]) |> real;
        # label = L"n_f(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
		line = (4.0, my_colors[:dω]),
		# xlabel = L"\omega / D",
		title = "Distribution function",
		legend = nothing,
		xlims = (-0.2, 0.2),
		ylims = (-0.01, 1.01),
		subplot = 2,
		titlefontsize = 16,
		guidefontsize = 16,
		# legendfontsize = 9,
		tickfontsize = 14,
		y_mirror = false,
		yticks = 0.5 * (0:1:2),
		xformatter = x -> x / scale,
		xticks = range(-0.2, 0.2, length=3),
		# right_margin = right_margin
    )

	plot!(
        ωs,
        (-GdL_ω[:, i] ./ ρd_ω[:, i]) |> real;
        # label = L"n_f(\omega, T=%$(round(Ts[i])))_{\tilde{\textrm{W}}}",
        line = (4.0, my_colors[:dτ]),
		subplot = 2,
    )
end

# ╔═╡ d3e5d08a-1201-4709-8b21-fb390b0ed674
begin
	labels = ["a", "b", "c", "d", "e", "f"]
	is = [i_t0 + 250, 3350, 3420, 3540, 3700, 4800]
	ps = [(right = iszero(mod(x, 2)); plot_gf(i, 
		right_margin = right ? 0Plots.mm : -15Plots.mm,
		bottom_margin = x < 5 ? -30Plots.mm : 0Plots.mm,
		xaxis = x < 5,
		y_mirror = right ? true : false,
		y_axis = right ? false : true,
		label = labels[x])) for (x, i) in enumerate(is)]

	pp = plot(
        Ts,
        nγ0,
		line = nothing,
		fill = (0, :grey, 0.7),
        label = nothing,
        xtick= params.tγ .- (-30:1:30) .* params.τK,
		xformatter = x -> (x - params.tγ) / params.τK,
		xlabel = L"t / \tau_K",
        legend = :topright,
		ylims = (0.0, 10.0),
		xlims = (30., 600),
		yticks = nothing
	)

	annotate!(45, 7.0, (L"(g)", 24), :black)

	
	for (x, i) in enumerate(is)
		pp = vline!([Ts[i]]; 
			label = nothing, 
			line = (:solid, 3.0, :black, 0.75)
		)
		annotate!(Ts[i] + 7., 7.8, (L"%$(labels[x])", 24), :black)
	end

	# pp = vspan!(pp,
 #        [600, 800];
 #        label = nothing,
 #        fillcolor = :red,
 #        alpha = 0.1,
 #        linewidth = 0,
 #        markersize = 0,
	# 	fillstyle = :/,
	# 	linecolor = :black
 #    )

	pp = plot(ps..., pp, layout = @layout [grid(3, 2){0.9h}; a{0.1h}])
	savefig("gd.svg")
end

# ╔═╡ 4b08f4dd-e9d0-4a40-8530-faadfd36da32
# function plot_spectral(i; ωmax = 1.0, clims = (0.0, 30), kwargs...)
# 	i_ω, GcR_k1, GdR_k1 = calculate_charge_carriers(i; ωmax = ωmax)

# 	xs = ωs[i_ω]
# 	ys = -(GdR_k1 + GcR_k1) |> imag

# 	h = heatmap(xs, ωs[i_ω], ys, 
# 		cmap=:viridis,
# 		xlabel = L"\kappa / \upsilon",
# 		clims = clims,
# 		colorbar = nothing, interpolate = true,
# 		margin = 4Plots.mm,
# 		kwargs...,
# 	)
# 	plot!(h, xs, xs, 
# 		xlims=extrema(xs), 
# 		ylims=extrema(xs), 
# 		line = (:dash, 1.0, :white, 0.3), 
# 		label = nothing)
# 	hline!(h, [0.0,], 
# 		line = (:solid, 1.0, :white, 0.3),
# 		label = nothing)
# 	annotate!(h, xs[1] * (1 - 0.7), xs[end] * (1 - 0.1), 
# 		L"T\ \upsilon=%$(round(Ts[i]))", :white)

# 	return h
# end

# ╔═╡ 7c708ebb-4ccc-4ae1-81eb-5cbb18462a18
begin
    function trace_path(k_path, n_points)
        _path(x0::Vector, Δx::Vector, n) = [x0 + Δx / n * i for i = 0:(n-1)]

        section_dist = map(norm, diff(k_path))
        section_length = n_points .* section_dist .÷ sum(section_dist) .|> Int

        path = mapreduce(_path, vcat, k_path, diff(k_path), section_length)
        dist = [0.0; cumsum(map(norm, diff(path)))]
        return path, dist
    end

	function calculate_charge_carriers(i_t; ωmax = 0.5)
		i_ω = findmin(abs.(ωs .+ ωmax))[2]:findmin(abs.(ωs .- ωmax))[2]

		# GcR_k = zeros(ComplexF64, length(ωs[i_ω]), length(ks))
		# GdR_k = zeros(ComplexF64, length(ωs[i_ω]), length(ks))
		GcR_k = zeros(ComplexF64, length(ωs[i_ω]), length(ωs[i_ω]))
		GdR_k = zeros(ComplexF64, length(ωs[i_ω]), length(ωs[i_ω]))
		_ΣcR = ΣcR_ω[i_ω, i_t]

		# for (i, ki) in collect(enumerate(ks))
		for (i, ωi) in enumerate(ωs[i_ω])
			for (j, ωj) in enumerate(ωs[i_ω])
				# GcR_k[j, i] = inv(ωj + 2sum(cos, ki) / 3 - _ΣcR[j])
				GcR_k[j, i] = inv(ωj - ωi - _ΣcR[j])
				GdR_k[j, i] = (_ΣcR[j] + _ΣcR[j] * GcR_k[j, i] * _ΣcR[j]) / abs2(params.V0)
			end
		end
		return i_ω, GcR_k, GdR_k
	end
end

# ╔═╡ bb2ffbe7-768f-4a07-982b-d686f6b2d12d
let
	scale = 2
	
	margins = Dict(
		:right_margin => -3Plots.mm, 
		:bottom_margin => -23Plots.mm, 
		:xticks => range(-0.8, 0.8, length=3), 
		:yticks => range(-0.8, 0.8, length=5)
	)

	annotate_size = 26
	
	i_ω, GcR_k1, GdR_k1 = calculate_charge_carriers(is[1]; ωmax = 1.0)
	_, GcR_k2, GdR_k2 = calculate_charge_carriers(is[3]; ωmax = 1.0)
	_, GcR_k3, GdR_k3 = calculate_charge_carriers(is[4]; ωmax = 1.0)
	_, GcR_k4, GdR_k4 = calculate_charge_carriers(is[6]; ωmax = 1.0)
	
	data1 = -(GdR_k1 + GcR_k1) |> imag
	data2 = -(GdR_k2 + GcR_k2) |> imag
	data3 = -(GdR_k3 + GcR_k3) |> imag
	data4 = -(GdR_k4 + GcR_k4) |> imag

	data1_2 = -(GdR_k1 + 0GcR_k1) |> imag
	data2_2 = -(GdR_k2 + 0GcR_k2) |> imag
	data3_2 = -(GdR_k3 + 0GcR_k3) |> imag
	data4_2 = -(GdR_k4 + 0GcR_k4) |> imag

	clims_1 = (0.0, 30)
	clims_2 = (0.0, 10)

	xs = ωs[i_ω]
	
	h1 = heatmap(xs, ωs[i_ω], data1, 
		cmap=:viridis,
		xlabel = L"\kappa / D",
		ylabel = L"\omega / D", 
		clims = clims_1,
		colorbar = nothing, 
		interpolate = true,
		xformatter = x -> x / scale,
		yformatter = x -> x / scale,
		xticks = margins[:xticks],
		yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		bottom_margin = margins[:bottom_margin],
		left_margin = 5Plots.mm
	)
	
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[1]))", annotate_size, :white))
	# annotate!(xs[1] * (1 - 0.7), xs[end] * (1 - 0.1), 
	# 	(L"t / \tau_K = %$(round((Ts[is[1]] - params.tγ) / params.τK, digits=2))", 10, :white))
	
	h2 = heatmap(xs, ωs[i_ω], data2, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		xaxis = nothing,
		# xlabel = L"\kappa / D",
		# xaxis = nothing,
		yticks = nothing,
		# xticks = nothing,
		xformatter = x -> x / scale,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_1,
		colorbar = nothing, interpolate = true,
		# xticks = margins[:xticks],
		# yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[3]))", annotate_size, :white))
	
	h3 = heatmap(xs, ωs[i_ω], data3, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		# xlabel = L"\kappa / D",
		# xaxis = nothing,
		yticks = nothing,
		xaxis = nothing,
		xformatter = x -> x / scale,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_1,
		colorbar = nothing, 
		interpolate = true, 
		# xticks = margins[:xticks],
		# yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[4]))", annotate_size, :white))

	h4 = heatmap(xs, ωs[i_ω], data4, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		# xlabel = L"\kappa / D",
		# xaxis = nothing,
		yticks = nothing,
		xformatter = x -> x / scale,
		# xticks = nothing,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_1,
		colorbar = nothing, 
		interpolate = true, 
		# xticks = margins[:xticks],
		xaxis = nothing,
		# yticks = margins[:yticks],
		# right_margin = margins[:right_margin],
		bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[6]))", annotate_size, :white))
	
	h5 = heatmap(xs, ωs[i_ω], data1_2, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		xlabel = L"\kappa / D",
		ylabel = L"\omega / D", 
		# title = "Charge carriers", 
		clims = clims_2,
		colorbar = nothing, interpolate = true,
		xformatter = x -> x / scale,
		yformatter = x -> x / scale,
		xticks = margins[:xticks],
		yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		# bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[1]))", annotate_size, :white))
	
	h6 = heatmap(xs, ωs[i_ω], data2_2, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		xlabel = L"\kappa / D",
		xformatter = x -> x / scale,
		yticks = nothing,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_2,
		
		colorbar = nothing, interpolate = true, 
		xticks = margins[:xticks],
		# yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		# bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[3]))", annotate_size, :white))
	
	h7 = heatmap(xs, ωs[i_ω], data3_2, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		xlabel = L"\kappa / D",
		xformatter = x -> x / scale,
		yticks = nothing,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_2,
		colorbar = nothing, interpolate = true, 
		xticks = margins[:xticks],
		# yticks = margins[:yticks],
		right_margin = margins[:right_margin],
		# bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[4]))", annotate_size, :white))

	h8 = heatmap(xs, ωs[i_ω], data4_2, 
		cmap=:viridis,
		# xticks = (ticks, hspath_scc_L), 
		xlabel = L"\kappa / D",
		xformatter = x -> x / scale,
		yticks = nothing,
		# ylabel = false,
		# ylabel = L"\omega / t", 
		# title = "Charge carriers", 
		clims = clims_2,
		colorbar = nothing, interpolate = true, 
		xticks = margins[:xticks],
		# yticks = margins[:yticks],
		# right_margin = margins[:right_margin],
		# bottom_margin = margins[:bottom_margin]
	)
	plot!(xs, xs, 
		xlims=extrema(xs), 
		ylims=extrema(xs), 
		line = (:dash, 1.0, :white, 0.2), 
		label = nothing)
	hline!([0.0,], 
		line = (:dash, 1.0, :white, 0.2),
		label = nothing)
	annotate!(xs[1] * (1 - 0.25), xs[end] * (1 - 0.1), 
		(L"(%$(labels[6]))", annotate_size, :white))
	
	# colorbar_1 = scatter([0], [0], zcolor=[NaN];
	#     xlims=(1,1),
	#     label=nothing,
	#     c=:viridis,
	#     clims=clims_1, 
	# 	framestyle=:none,
	# 	bottom_margin = margins[:bottom_margin]
	# 	# top_margin = 10Plots.mm
	# )
	# colorbar_2 = scatter([0], [0], zcolor=[NaN];
	#     xlims=(1,1),
	#     label=nothing,
	#     c=:viridis,
	#     clims=clims_2, 
	# 	framestyle=:none,
	# 	# left_margin = 10Plots.mm,
	# 	# top_margin = 5Plots.mm
	# )

	blank1 = plot(
		# foreground_color_subplot=:white, 
		inset = (1, bbox(0.0, 0.07, 1.0, 0.85)), 
		showaxis = false, 
		bottom_margin = margins[:bottom_margin],
	)
	
	scatter!([0], [0], zcolor=[NaN];
	    xlims=(1,1),
	    label=nothing,
	    c=:viridis,
	    clims=clims_1, 
		framestyle=:none,
		# background_color_subplot=:transparent, 
		subplot=2,
	)

	blank2 = plot(
		# foreground_color_subplot=:white, 
		inset = (1, bbox(0.0, 0.07, 1.0, 0.85)), showaxis = false
	)
	
	scatter!([0], [0], zcolor=[NaN];
	    xlims=(1,1),
	    label=nothing,
	    c=:viridis,
	    clims=clims_2, 
		framestyle=:none,
		# background_color_subplot=:transparent, 
		# markerstrokecolor=:transparent,
		subplot=2,
		# bottom_margin = margins[:bottom_margin]
	)
	
	l = @layout [grid(2,4) grid(2,1){0.05w}]
	
	plot(h1, h2, h3, h4, h5, h6, h7, h8, blank1, blank2, link=:all, layout = l)
	
	
	
	# plot(h1, h2, h3, h4, h5, h6, h7, h8, colorbar_1, colorbar_2, 
	# 	layout = @layout [grid(2,4) grid(2,1){0.05w}])

	savefig("collapse-revival-heatmap.svg");
end

# ╔═╡ 69580e8e-38cf-4f89-abb7-724bf482e06f
# let
#     Plots.gr()

#     Gd_τ = let
# 		τs, ρd_τ = KadanoffBaym.ft(ωs, ρd_ω[:, i_t0]; inverse = true)
#         plot(
#             τs,
#             -imag(ρd_τ);
#             label = L"-2\,\operatorname{Im}\,G^R_d(\tau, T=%$(round(Ts[i_t0])))_{\textrm{W}}",
#             title = "Spectral functions",
#             xlabel = L"\tau \;D",
#             xlims = params.τK .* (-2 / 3, 2 / 3),
#             legend = :topright,
#             alpha = 1.0,
#             ms = 1.0,
#             linecolor = my_colors[:dω],
#             left_margin = 5Plots.mm,
#             right_margin = 15Plots.mm,
#         )
#     end

#     Gd_τ = let
# 		τs, ρd_τ = KadanoffBaym.ft(ωs, ρd_ω[:, i_max_pulse]; inverse = true)
# 		plot!(
# 	        τs,
# 	        -imag(ρd_τ);
# 	        label = L"-2\,\operatorname{Im}\,G^R_d(\tau, T=%$(round(Ts[i_max_pulse])))_{\textrm{W}}",
# 	        ms = 2.0,
# 	        linecolor = my_colors[:dτ],
# 	    )
# 	end

#     Gd_τ_kondo = let
#         # Fourier transform to τ-space just the peak
#         kondo_ft = KadanoffBaym.ft(ωs[i1:i2], -ρd_ω[i1:i2, i_t0]; inverse = true)

#         Gd_τ_kondo = plot(
#             kondo_ft[1],
#             imag(kondo_ft[2]);
#             label = L"-2\,\operatorname{Im}\,G^{R}_d(\tau)",
#             xlabel = L"\tau \;D",
#             linecolor = my_colors[:dω],
#             legend = :topright,
#             xlims = params.τK .* (-10, 10),
#             title = "Kondo peak duration",
#         )

# 		# plot!(
# 	 #        Gd_τ_kondo,
# 	 #        kondo_ft[1],
# 		# 	maximum(imag(kondo_ft[2])) * exp.(-abs.(kondo_ft[1]) / params.τK);
# 	 #        # fillcolor = my_colors[:ωK],
# 	 #        # alpha = 0.3,
# 	 #        # label = L"\tau_K",
# 	 #        # linewidth = 0,
# 	 #        # markersize = 0,
# 	 #    )
		
#         plot!(
#             Gd_τ_kondo,
#             [],
#             [];
#             label = L"-2\,\operatorname{Im}\,G^{0,R}_\gamma(\tau)",
#             linecolor = my_colors[:ω0],
#         )

# 		τs, ργ0_τ = KadanoffBaym.ft(ωs, ργ0_ω[:, i_max_pulse]; inverse = true)
#         plot!(
#             twinx(),
#             τs,
#             -imag(ργ0_τ);
#             linecolor = my_colors[:ω0],
#             xaxis = nothing,
#             legend = nothing,
#         )

#         scatter!(
#             ωs[i1:i2],
#             -imag(ρd_ω[i1:i2, i_t0_]);
#             xlabel = L"\omega / D",
#             markersize = 2.0,
#             marker = (1.0, :grey),
#             xtickfontsize = 5,
#             ytickfontsize = 5,
#             xguidefontsize = 5,
#             label = nothing,
#             inset = (1, bbox(0.05, 0.1, 0.3, 0.4)),
#             subplot = 3,
#         )
#     end

#     vspan!(
#         Gd_τ_kondo,
#         params.τK * [-1 / 2, 1 / 2];
#         fillcolor = my_colors[:ωK],
#         alpha = 0.3,
#         label = L"\tau_K",
#         linewidth = 0,
#         markersize = 0,
#     )

#     plot(Gd_τ, Gd_τ_kondo; layout = grid(2, 1))
#     # savefig("equilibrium-comparison.pdf")
# end

# ╔═╡ 0a29a330-9762-44da-8dbb-bdf2f52dfe0b
# let
#     ylims_ργ = (0.0, max(maximum(nγ), maximum(nγ0)))
#     ylims_ρd = (0.0, maximum(-imag(ρd_ω)))

# 	kondo_average_number = map(x -> ∫dωK(x)[1], eachcol(imag(GdL_ω)))
# 	kondo_maximum_number = map(x -> maximum(x[i1:i2]), eachcol(imag(GdL_ω)))
# 	kondo_average_spectral = map(x -> ∫dωK(-x)[1], eachcol(imag(ρd_ω)))
# 	kondo_maximum_spectral = map(x -> maximum(-x[i1:i2]), eachcol(imag(ρd_ω)))
	
#     function plotter(i_t, i_max)
# 		h1 = plot(
#         ωs,
#         -imag(ρd_ω[:, i_t0_]);
#         label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
#         xlabel = L"\omega / t",
#         xlims = (-1.0, 1.0),
#         ylims = (0.0, Inf),
#         legend = :topleft,
# 		legendfontsize = 25,
#         line = (5.0, my_colors[:dω]),
#         title = "Kondo peak melt",
# 		titlefontsize = 60,
# 		xtickfont = 30,
# 		ytickfont = 30,
# 		xguidefontsize = 40,
# 		margin = 15Plots.mm
#     )

#     h1 = plot!(
#         ωs,
#         imag(GdL_ω[:, i_t0_]);
#         label = nothing,
#         fill = (0, my_colors[:dω], 0.5),
#         linecolor = nothing,
#     )

#     h1 = plot!(
#         ωs,
#         -imag(ρd_ω[:, i_t]);
#         label = L"-2\,\operatorname{Im}\,G^R_d(\omega, T=%$(round(Ts[i_t])))_{\tilde{\textrm{W}}}",
#         line = (5.0, my_colors[:dτ]),
#     )

#     h1 = plot!(
#         ωs,
#         imag(GdL_ω[:, i_t]),
#         label = nothing,
#         fill = (0, my_colors[:dτ], 0.5),
#         linecolor = nothing,
#     )

#     # h1 = vline!([params.ε0f]; label = L"\varepsilon_d", line = (:dot, 5.0, my_colors[:εd], 0.75))

#     h1 = vline!(
#         [-params.ω0, params.ω0];
#         label = L"\omega_0",
#         line = (:dot, 5.0, my_colors[:ω0], 0.75),
#     )

# 	h1 = plot!(
#         ωs,
#         (-GdL_ω[:, i_t0_] ./ ρd_ω[:, i_t0_]) |> real;
#         label = L"f_d(\omega, T=%$(round(Ts[i_t0])))_{\tilde{\textrm{W}}}",
# 		line = (5.0, my_colors[:dω]),
#         # title = "Distribution functions",
# 		xlabel = L"\omega / t",
# 		legend=:bottomleft,
# 		xlims = (-0.15, 0.15),
# 		ylims = (-0.1, 1.1),
# 		inset = (1, bbox(0.63, 0.23, 0.35, 0.55)),
# 		subplot = 2,
# 		titlefontsize = 30,
# 		legendfontsize = 20,
# 		xtickfont = 30,
# 		ytickfont = 30,
# 		xguidefontsize = 30
#     )

# 	h1 = plot!(h1,
#         ωs,
#         (-GdL_ω[:, i_t] ./ ρd_ω[:, i_t]) |> real;
#         label = L"f_d(\omega, T=%$(round(Ts[i_t])))_{\tilde{\textrm{W}}}",
#         line = (5.0, my_colors[:dτ]),
# 		subplot = 2,
#     )
# 		# PHOTON OCCUPATION
		
#         # sa(t) = [
#         #     range(0, 1; length = max(t - 5, 2))
#         #     ones(max(1, min(5, t)))
#         #     zeros(length(ts) - t)
#         # ]
#         # sa(t) = range(0, 1; length=t)
#         # sa(t) = let
#         # 	v = max(1, min(5,t))
#         # 	[
#         # 		t-v == 1 ? [0.0] : range(0, 1; length=max(t-v, 0))
#         # 		ones(v)
#         # 	]
#         # end
#         sa(t) =
#             let
#                 t = t - i_t0 + 1
#                 v = max(1, min(5, t))
#                 [
#                     1 # needed to get the correct label marker
#                     t - v == 1 ? [0.0] : range(0, 1; length = max(t - v, 0))
#                     ones(v - 1)
#                 ]
#             end

#         # Occupations
#         p1 = plot(
#             Ts[i_t0:i_t0],
#             nγ0[i_t0:i_t0];
#             label = L"n^\mathrm{in}_\gamma",
# 			xlabel = L"T\;t",
#             line = (5.0, :grey),
#             grid = false,
#             seriesalpha = 0.8sa(i_t),
#             ylims = ylims_ργ,
#             xlims = (Ts[i_t0], Ts[i_max]),
#             legend = :bottomleft,
# 			legendfontsize = 25,
# 			margin = 3Plots.mm,
# 			xtickfont = 30,
# 			ytickfont = 30,
# 			xguidefontsize = 40
#         )

#         p1 = plot!(
#             Ts[i_t0:i_t0],
#             nγ[i_t0:i_t0];
#             line = (5.0, my_colors[:ω0]),
#             label = L"\Delta n_\gamma^\mathrm{out}",
#         )

# 		p1 = vline!([Ts[i_t]]; label = nothing, line = (:dot, 5.0, :grey, 0.75))

# 		p1 = plot!(
#             Ts[i_t0:i_t0],
#             nγ0[i_t0:i_t0];
#             label = L"n^\mathrm{in}_\gamma",
# 			xlabel = L"T\;t",
#             line = fill = (0, :grey, 0.5),
#             seriesalpha = 0.8sa(i_t),
			
#     		# legend = nothing,
			
#             xlims = (Ts[i_t0], Ts[i_max]),
# 			# ylims = (0.0, 0.1),
			
# 			inset = (1, bbox(0.63, 0.05, 0.35, 0.75)),
# 			titlefontsize = 10,
# 			legendfontsize = 10,
# 			xtickfont = 30,
# 			ytickfont = 30,
# 			# xguidefontsize = 40
# 			subplot = 2,
#         )

#         p1 = plot!(
#             Ts[i_t0:i_t0],
#             nγ_vac[i_t0:i_t0];
#             line = (my_colors[:ω0], 5.0),
#             label = L"n_\gamma^\mathrm{out}",
# 			subplot = 2,
#         )

# 		p1 = vline!([Ts[i_t]]; label = nothing, line = (:dot, 5.0, :grey, 0.75), subplot = 2)
		

# 		plt = twinx()
		
#         p1 = plot!(
#             # plt,
#             Ts[i_t0:i_t],
#             nγ0[i_t0:i_t];
#             # line = (my_colors[:c], 5.0, 0.5),
# 			fill = (0, :grey, 0.5),
# 			label = nothing,
# 			# legend = nothing,
# 			# yaxis = nothing,
# 			# xaxis = nothing,
# 			# grid = nothing,
			
#             xlims = (Ts[i_t0], Ts[i_max]),
#             ylims = ylims_ργ,
#             seriesalpha = 0.8sa(i_t),
#         )

#         p1 = plot!(
#             Ts[i_t0:i_t],
# 			100000nγ_vac[i_t0:i_t];
#             # nγ_vac[i_t0:i_t];
#             label = nothing,
#             line = (my_colors[:ω0], 5.0, 0.8),
#             seriesalpha = 0.8sa(i_t),
#         )

#         p1 = plot!(
#             Ts[i_t0:i_t],
#             nγ0[i_t0:i_t];
#             label = nothing,
#             # line = (my_colors[:c], 5.0, 0.5),
# 			fill = (0, :grey, 0.5),
#             # legend = :topright,
# 			# label = nothing,
# 			legend = nothing,
#             # xlabel = L"T\;D",
#             xlims = (Ts[i_t0], Ts[i_max]),
#             ylims = (-0.01, 0.05),
#             grid = false,
#             seriesalpha = 0.8sa(i_t),
#             # margin = 5Plots.mm,
# 			subplot = 2,
# 			xtickfont = 30,
# 			ytickfont = 30,
# 			xguidefontsize = 30,
#         )

#         p1 = plot!(
#             Ts[i_t0:i_t],
# 			10000nγ_vac[i_t0:i_t];
#             #nγ_vac[i_t0:i_t];
#             label = nothing,
#             line = (my_colors[:ω0], 5.0, 0.8),
#             seriesalpha = 0.8sa(i_t),
# 			subplot = 2,
#         )
		
# 		# plot!(Ts[i_t0:i_t], 300kondo_average_number[i_t0:i_t] .+ 0,
# 		# 	label = "Avg Kondo-particle-number",
# 		# 	line = (my_colors[:dτ], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)

# 		# plot!(Ts[i_t0:i_t-1], 4000diff(kondo_average_number[i_t0:i_t]) ./ diff(Ts[i_t0:i_t]) .+ 10, 
# 		# 	label = "Deriv. Avg Kondo-particle-number",
# 		# 	line = (my_colors[:ω0], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)
		
# 		# plot!(Ts[i_t0:i_t], 3kondo_maximum_number[i_t0:i_t] .- 5,
# 		# 	label = "Max Kondo-particle-number",
# 		# 	line = (my_colors[:dω], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)

# 		# plot!(Ts[i_t0:i_t-1], 60diff(kondo_maximum_number[i_t0:i_t])  ./ diff(Ts[i_t0:i_t]) .+ 10, 
# 		# 	label = "Deriv Max Kondo-particle-number",
# 		# 	line = (my_colors[:ω0], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)
		
# 		# plot!(Ts[i_t0:i_t], 100kondo_average_spectral[i_t0:i_t], 
# 		# 	label = "Avg Kondo-spectral-weight",
# 		# 	line = (my_colors[:ω0], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)
	
# 		# plot!(Ts[i_t0:i_t], kondo_maximum_spectral[i_t0:i_t], 
# 		# 	label = "Max Kondo-spectral-weight",
# 		# 	line = (my_colors[:εd], 5.0, 0.5),
# 		# 	# margin=10Plots.mm, 
# 		# 	# xlims = (0, Inf),
# 		# 	# xformatter = x -> string(floor(Int, x * params.ωK))
# 		# 	)

# 		# # Kondo time-scale
#   #       p1 = vspan!(
#   #           p1,
#   #           params.tγ .+ params.τK .* [-0.5, +0.5];
#   #           fillcolor = :grey,
#   #           alpha = 0.1,
#   #           linewidth = 0,
#   #           markersize = 0,
#   #           label = L"\tau_K",
#   #       )
		
# 		# p1 = vspan!(
#   #           params.tγ .+ params.τK .* [-0.5, +0.5];
#   #           fillcolor = :grey,
#   #           alpha = 0.1,
#   #           linewidth = 0,
#   #           markersize = 0,
# 		# 	subplot = 2
#   #       )

#         return plot(h1, p1; size = 4 .* (600, 450), layout = @layout grid(2, 1))
#     end

#     anim = @animate for t in range(i_t0, i_t0 + 3000, step=4)#range(i_t0, length(Ts), step = 4)
#         iszero(mod(t, 10)) ? (@info Ts[t]) : nothing
#         plotter(t, i_t0+3000)
#     end

#     gif(anim, "anim_fps15.mp4", fps = 24)

# 	# plotter(i_t0+2000, i_t0+3000)
# end;

# ╔═╡ Cell order:
# ╟─0a46b1e2-b862-495d-a069-76ac6b27b556
# ╠═f5dbf536-247b-11ed-1663-2134cf17d1fb
# ╠═95e5d72e-aaa8-49fa-a473-d96507f1ba7e
# ╠═0190be17-7d73-4522-8cc3-d1500801c408
# ╠═1efa1af7-bd2d-49c2-8f4a-4cb65fc63255
# ╟─330f7766-daf8-4d47-a38a-b7797b5e6264
# ╠═c5b0d68b-0b1b-4795-ac59-521988dff946
# ╠═cec8ba06-98c8-48a8-a360-1b19cda1e9de
# ╟─e375a68f-a9d1-4104-8548-0b3858cc9d32
# ╠═712bac76-feba-44f7-a094-093e7f45a20f
# ╠═63e379c3-bfab-4e88-ba07-c58b6ff7fff6
# ╠═d3e5d08a-1201-4709-8b21-fb390b0ed674
# ╠═4b08f4dd-e9d0-4a40-8530-faadfd36da32
# ╠═7c708ebb-4ccc-4ae1-81eb-5cbb18462a18
# ╠═bb2ffbe7-768f-4a07-982b-d686f6b2d12d
# ╟─69580e8e-38cf-4f89-abb7-724bf482e06f
# ╟─0a29a330-9762-44da-8dbb-bdf2f52dfe0b
