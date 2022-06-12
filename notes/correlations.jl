### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 1cf881a4-ea46-11ec-3c1c-73c3deb8f624
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(m, Expr(:toplevel, :(eval(x) = $(Expr(:core, :eval))($name, x)), :(include(x) = $(Expr(:top, :include))($name, x)), :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)), :(include($path))))
    m
end;

# ╔═╡ efd2fee3-ba72-42f5-9e7f-fe32eee11a0d
begin
    using Pkg
    Pkg.activate()

    using LinearAlgebra: diag
    using KadanoffBaym, LangrethsRules

    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Plots, LaTeXStrings
end;

# ╔═╡ fb07d4eb-95c0-497a-8fe5-8515026b5b89
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
		:ω0 => Plots.Colors.colorant"#FF2C00"
		)
end;

# ╔═╡ 15bb008a-6b89-4cd6-aad5-5c30da31a7a5
# parameters (units of half-bandwidth)
begin
    # inverse temperature
    β = 1e2

    # model parameters
    model = EchoPulse.PhotonAssistedAIM(
        # siam
        ;
        V0 = sqrt(0.1),# hybridization strength
        ε0f = -0.35,# d-electron energy
        U0 = Inf,

        # quantum light
        Γ0 = 0sqrt(0.01), # light-matter coupling
        ω0 = 8.0e-2, # photon energy
        Ω0 = 1.5 * 8.0e-2, # photon frequency bandwidth
    )

    # pulse time parameters
    Nγ = 5.0e-0 # maximum photon occupation
    Tγ = 1.0e2 # peak arrival time

    # Bethe lattice dmft
    dmft = true

    # solve with time-matrix initial condition
    timematrix = true

    # integration parameters
    tmax = 1e2
    rtol = 1e-5
end;

# ╔═╡ 75af13de-fcb5-4e1d-804f-dc3cc0809eed
begin
    ωs_eq = range(-7.0, 7.0; length = 2^15 - 1)

    (; data, GdRω, GcRω, 𝒢cRω, ωK, t0) = EchoPulse.PhotonAssistedAIMData(model = model, ωs = ωs_eq, β = β, Tγ = Tγ, Nγ = Nγ, timematrix = timematrix, dmft = dmft)
end;

# ╔═╡ 21895ece-5cf4-4f6f-b851-44d4fefb1788
begin
    cecu6_half_bandwidth = 2.0 # eV
    cdos_half_bandwidth = 2
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth

    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 3)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)

    τK = inv(ωK)

    md"""
    #### Summary
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the  parameters are
    1. ``\varepsilon_d =`` $(ω_to_eV(model.ε0f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(model.V0)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(ωK)) ``ps`` (Kondo temperature & timescale)
    4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
    and the gaussian-shaped pulse parameters were
    5. ``\omega = `` $(ω_to_THz(model.ω0)) ``THz`` (pulse energy)
    6. ``\Omega = `` $(ω_to_THz(model.Ω0)) ``THz``  (pulse bandwidth: $(ω_to_ps(model.Ω0/6)) ``ps``)
    7. ``\Gamma = `` $(ω_to_eV(model.Γ0)) ``eV`` (light-matter coupling)
    8. ``N_\gamma = `` $(round(Nγ, digits=3)) ``n_\gamma`` (number of photons)

    The system will be integrated in the interval (0, $(ω_to_ps(inv(tmax)))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(Tγ))) ``ps``.
    """
end

# ╔═╡ 1ecc4622-b311-47bd-8ba0-d6effa01fa35
let	
    plot(ωs_eq, -imag(GcRω); 
		label = L"-\textrm{Im}\sum_k G^R_{c}(k,\omega)", 
		line = (my_colors[:c], ),
		xlabel = L"\omega/D", 
		title = "Equilibrium spectral functions", 
		xlims = 0.5 .* extrema(ωs_eq), 
		ylims = (0.0, Inf)
	)

    plot!(ωs_eq, -imag(GdRω); 
		label = L"-\textrm{Im}G^R_d(\omega)", 
		line = (my_colors[:dω], ),
	)

    vline!(model.ε0f * [1.0]; 
		label = L"\varepsilon_d", 
		line = (:dot, 1.5, my_colors[:εd], 0.75)
	)

    vline!(model.ω0 * [-1.0, 1.0]; 
		label = L"\omega_0", 
		line = (:dot, 1.5, my_colors[:ω0], 0.75)
	)

    vspan!(ωK * [-1 / 2, 1 / 2]; 
		label = L"\omega_K",
		fill = (my_colors[:ωK], 0.3),
	)
end

# ╔═╡ d12b32d7-4101-44c9-87b1-4ed1438decf7
begin
    # Time-step
    @time ts, ws = kbsolve!(
		(x...) -> EchoPulse.fv!(x..., model, data; Tγ = Tγ, pulse = true), 
        (x...) -> EchoPulse.fd!(x..., model, data; Tγ = Tγ, pulse = true), 
        EchoPulse.initial_condition(data; dmft = dmft), 
        (t0, tmax); 
        dtini = 1e-6, 
        rtol = rtol, 
        atol = 1e-3 * rtol, 
        callback = (x...) -> begin
            EchoPulse.resize_Σ!(data)
            EchoPulse.calculate_Σt!(x..., model, data; dmft = dmft)
        end, 
        stop = (x...) -> begin
            println(" t: $(round(last(x[1]), digits=3))")
            false
        end)

    # Post processing
    EchoPulse.resize_Σ!(data; resize = true)

    # Calculate observables
    (; Gd, Gc, G0γ, Gγ, Gγ_vac) = EchoPulse.project(data, model; ts = ts, ws = ws, dmft = dmft)
end;

# ╔═╡ 5feec3dd-f53d-4e67-8b1a-ca66131b17eb
begin
    # Calculate optimal grid length
    ol = let
        # err = map(range(8length(ts) ÷ 10, 12length(ts) ÷ 10, step=1)) do y
        # 	map(range(first(ts), last(ts), length=y)) do x
        # 		minimum(abs.(ts .- x))
        # 	end |> x -> sum(x) / y
        # end

        length(ts)
        # 8length(ts) ÷ 10 + argmin(err) - 1
    end

    # Spectral function
    ρd_t = (greater(Gd) - lesser(Gd))
    ρc_t = (greater(Gc) - lesser(Gc))
    ργ_t = (greater(G0γ) - lesser(G0γ))

    ρd_τ, (τs, Ts) = wigner_transform_itp(ρd_t, ts; fourier = false)#, len=ol)
    ρd_ω, (ωs, Ts) = wigner_transform_itp(ρd_t, ts; fourier = true)#, len=ol)
    ρc_ω, _ = wigner_transform_itp(ρc_t, ts; fourier = true)#, len=ol)
    ργ_τ, _ = wigner_transform_itp(ργ_t, ts; fourier = false)#, len=ol)

	GdL_ω, _ = wigner_transform_itp(lesser(Gd), ts; fourier = true)
end;

# ╔═╡ 59d727ac-5610-4443-9783-f1f4b69d863a
let
    # Spectral function
    h1 = heatmap(
        Ts,
        ωs,
        -imag(ρd_ω);
        title = L"A_d(\omega,T)",
        xaxis = nothing,
        ylabel = L"\omega /D",
        minorticks = true,
        tick_direction = :out,
        ylims = (-min(2.0, abs(minimum(ωs))), min(2.0, maximum(ωs))),
        legend = :none,
        c = :magma,
        rightmargin = 5Plots.mm,
    )

    colorbar = heatmap(
        range(0, 1, length = length(ωs)) .* ones(1, 1),
        legend = nothing,
        xaxis = false,
        # yaxis = false,
        label = nothing,
        grid = nothing,
        c = h1.subplots[1][1][:linecolor],
        xticks = false,
        yticks = (Plots.get_colorbar_ticks(h1.subplots[1])[1] * length(ωs) / h1.subplots[1][1][:clims_calculated][2], Plots.get_colorbar_ticks(h1.subplots[1])[2]),
        tick_direction = :out,
        ymirror = true,
    )

    # Occupations
    p1 = plot(
        ts,
        imag(diag(data.GfL.data));
        label = L"n_f",
        title = "Field occupations",
        legend = :topleft,
        xlims = extrema(ts),
        minorticks = true,
        # tick_direction = :out,
        xlabel = L"T\;D",
        rightmargin = 5Plots.mm,
        # minorgrid=true
    )
	
    p1 = plot!(ts, imag(diag(-data.GbL.data)); 
    	label = L"n_b"
	)

	p1 = plot!(
		ts,
		imag(diag((2data.GfL.data - data.GbL.data - data.GaL.data)));
		label = L"Q",
		line = (:dot, :grey, 1),
	)

    # Kondo timescale
    # vspan!(
    #        Tγ .+ τK * [-1/2, 1/2];
    #        fillcolor = my_colors[:ωK],
    #        alpha = 0.3,
    #        label = L"\tau_K",
    # 	linewidth = 0,
    #        markersize = 0,
    # 	# xaxis = nothing,
    #    )

    # twinx_ = twinx()

    # THz pulse
    #    p1 = plot!(
    #        twinx_,
    #        ts,
    # 	G0γ |> (-) ∘ imag ∘ diag ∘ lesser,
    #        # G0γ |> (-) ∘ imag ∘ diag ∘ lesser;
    #        label = L"n^0_\gamma",
    #        line = (:green, 3.0, 0.2),
    #        legend = :topright,
    #        xlims = extrema(ts),
    #        grid = false,
    #        minorticks = true,
    #        tick_direction = :out,
    #        xaxis = nothing,
    #    )
    # p1 = plot!(
    #        twinx_,
    #        ts,
    #        Gγ |> (-) ∘ imag ∘ diag ∘ lesser;
    #        label = L"n_\gamma",
    #        line = (:darkgreen, 1.0),
    #    )
    # p1 = plot!(
    #        twinx_,
    #        ts[2:end],
    #        diff(Gγ |> (-) ∘ imag ∘ diag ∘ lesser) ./ diff(ts);
    #        label = L"n_\gamma",
    #        line = (:darkgreen, 1.0),
    #    )

    plot(h1, p1, colorbar; size = (600, 450) .* 1.0, layout = @layout [grid(2, 1){0.95w} a])
end

# ╔═╡ 09f25d8b-7d87-45f7-9a42-68a3027485cf
let
    plot(
        ts,
        -imag(diag(lesser(G0γ))),
        label = L"n_\gamma^\mathrm{in}",
        xlabel = L"T\;D",
        line = (:darkgreen),
        ylims = (0.0, Nγ),
        # legend=:right,
        left_margin = 5Plots.mm,
        right_margin = 15Plots.mm,
        title = "Photon output",
    )

    plot!([], [], 
        label = L"n_\gamma^\mathrm{out} - n_\gamma^\mathrm{in}", 
        color = :orange
    )

    vspan!(Tγ .+ τK .* [-0.5, +0.5]; 
        label = L"\tau_K", 
        fillcolor = :grey, 
        alpha = 0.1, 
        linewidth = 0, 
        markersize = 0
    )

    i0 = findmin(x -> abs(x), ts)[2]

    # Gγ = -imag(diag((lesser(Gγ_out))))
    # # Gγ = -imag(diag((lesser(Gγ_out) - lesser(G0γ))))
    # plot!(twinx(), ts[i0:end], Gγ[i0:end];
    # 	color=:blue,
    # 	label=nothing, 
    # 	ymirror = true,
    # 	# ylims=(Gγ[end], maximum(Gγ))
    # )

	# Gγ = -imag(diag(lesser(Gγ)))
    Gγ = -imag(diag(lesser(Gγ_vac)))
    plot!(
        twinx(),
        ts[i0:end],
        Gγ[i0:end];
        color = :orange,
        label = nothing,
        ymirror = true,
        # ylims=(Gγ[end], maximum(Gγ))
    )

    xlims!(0.0, ts[end])
    # ylims!(0.0, 0.1)
    # savefig("test.pdf")
end

# ╔═╡ a0ebd9bf-29a6-48c2-ae33-3aa8a25666ca
# Comparison of frequency *projected* solution with time solution
let
	Plots.gr()
	
    plot(ωs_eq, -2imag(GdRω); 
        label = L"-2\operatorname{Im}\,G^R_d(\omega)", 
        xlabel = L"\omega / D", 
        xlims = (ωs[1], ωs[end]), 
        legend = :topright, 
        linecolor = my_colors[:dω], 
        title = "Kondo peak melt", 
        ylims = (0.0, Inf)
    )
    plot!(
           ωs_eq,
           -2imag(GdRω) .* EchoPulse.nf.(ωs_eq, β);
           label = nothing,
           fill=(0,my_colors[:dω],0.5),
           linecolor = nothing,
       )

    max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]

    plot!(ωs, -imag(ρd_ω[:, max_pulse_i]); 
        label = L"-2\operatorname{Im}\,G^R_d(T_0,\omega)_{\tilde{\textrm{W}}}", 
        linecolor = my_colors[:dτ]
    )

    plot!(ωs, imag(GdL_ω[:, max_pulse_i]),
        label = nothing,#L"\mathcal{F}_\tau\{A^<_d(T_0,\tau)\}(\omega)",
        fill = (0, my_colors[:dτ], 0.5),
        linecolor = nothing,
    )

    vline!([model.ε0f]; 
        label = L"\varepsilon_d", 
        line = (:dot, 1.5, my_colors[:εd], 0.75)
    )

    vline!([-model.ω0, model.ω0]; 
        label = L"\omega_0", 
        line = (:dot, 1.5, my_colors[:ω0], 0.75)
    )

    vspan!(ωK * [-1 / 2, 1 / 2]; 
        fillcolor = my_colors[:ωK], 
        alpha = 0.3, 
        label = L"\omega_K", 
        linewidth = 0, 
        markersize = 0
    )

    xlims!(-1.0, 1.0)
end

# ╔═╡ 0cd1b4c4-2e56-4027-95b5-24f801f5c04c
let
    Gd_τ = let
        f = EchoPulse.ft(ωs_eq, -2im * imag(GdRω); inverse = true)
        plot(f[1], imag(f[2]); 
            label = L"-2\operatorname{Im}\,G_d(\tau)", 
			title = "Kondo peak duration",
            xlabel = L"\tau \;D", 
            xlims = τK .* (-2 / 3, 2 / 3), 
            legend = :topright, 
            alpha = 1.0, ms = 1.0, 
            linecolor = my_colors[:dω], 
            left_margin = 5Plots.mm, 
            right_margin = 15Plots.mm
        )
    end

    max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]

    Gd_τ = plot!(τs, -imag(ρd_τ[:, max_pulse_i]); 
        label = L"-2\operatorname{Im}\,G_d(T_0,\tau)", 
        ms = 2.0, linecolor = 
        my_colors[:dτ]
    )

    Gd_τ_kondo = let
        # indices of the "Kondo peak"
        w = 10
        i1 = argmin(abs.(ωs_eq .+ w * ωK))
        i2 = argmin(abs.(ωs_eq .- w * ωK))

        # Fourier transform to τ-space just the peak
        kondo_ft = EchoPulse.ft(ωs_eq[i1:i2], -2im * imag(GdRω)[i1:i2]; inverse = true)

        Gd_τ_kondo = plot(kondo_ft[1], imag(kondo_ft[2]); 
            label = L"-2\operatorname{Im}\,G^{\textrm{Kondo}}_d(\tau)", 
            xlabel = L"\tau \;D", 
            linecolor = my_colors[:dω], 
            xlims = τK .* (-2 / 3, 2 / 3)
        )

        plot!(Gd_τ_kondo, [], []; 
            label = L"-2\operatorname{Im}\,G_\gamma(T_0,\tau)", 
            linecolor = my_colors[:ω0]
        )

        plot!(twinx(), τs, -imag(ργ_τ[:, max_pulse_i]); 
            linecolor = my_colors[:ω0], 
            xaxis = nothing, 
            legend = nothing, 
            xlims = inv(ωK) .* (-2 / 3, 2 / 3)
        )
    end

    vspan!(
        Gd_τ_kondo,
        τK * [-1 / 2, 1 / 2];
        fillcolor = my_colors[:ωK],
        alpha = 0.3,
        label = L"\tau_K",
        linewidth = 0,
        markersize = 0,
        # xaxis = nothing,
    )

    plot(Gd_τ, Gd_τ_kondo; layout = grid(2, 1))
    # savefig("equilibrium-comparison.pdf")
end

# ╔═╡ 96b7976d-5e96-4031-8f13-81f1fe14bfc3
# let
# 	# indices of the "Kondo peak"
# 	w = 10
# 	i1 = argmin(abs.(ωs_eq .+ w*ωK))
# 	i2 = argmin(abs.(ωs_eq .- w*ωK))

# 	max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]

# 	Gd_ω = scatter(ωs_eq[i1:i2], -2im * imag(GdRω)[i1:i2] |> imag)

# 	Gd_τ_kondo = let
# 		# Fourier transform to τ-space just the peak
# 		kondo_ft = EchoPulse.ft(ωs_eq[i1:i2], -2im * imag(GdRω)[i1:i2]; 
# 			inverse = true)

# 		Gd_τ_kondo = plot(kondo_ft[1], imag(kondo_ft[2]);
# 			label = L"A^{(kondo)}_d(\tau)",
# 			xlabel = L"\tau \;D",
# 			linecolor = my_colors[:dω],
# 			xlims=τK .* (-2/3, 2/3)
# 			)

# 		plot!(Gd_τ_kondo, [], [];
# 			label = L"A_\gamma(T_0,\tau)",
# 			linecolor = my_colors[:ω0],
# 		)

# 		plot!(twinx(), τs, -imag(ργ_τ[:, max_pulse_i]);
# 			label = L"A_\gamma(T_0,\tau)",
# 			linecolor= my_colors[:ω0],
# 			xaxis = nothing,
# 			legend = nothing,
# 			xlims=inv(ωK) .* (-2/3, 2/3)
# 		)
# 	end

# 	vspan!(
# 		Gd_τ_kondo,
#         τK * [-1/2, 1/2];
#         fillcolor = my_colors[:ωK],
#         alpha = 0.3,
#         label = L"\tau_K",
# 		linewidth = 0,
#         markersize = 0,
# 		# xaxis = nothing,
#     )

# 	plot(Gd_ω, Gd_τ_kondo; layout = grid(2, 1))
#     # savefig("equilibrium-comparison.pdf")
# end

# ╔═╡ 364199d9-a3f6-4f0f-8f63-60420b9da2c2


# ╔═╡ Cell order:
# ╟─1cf881a4-ea46-11ec-3c1c-73c3deb8f624
# ╟─efd2fee3-ba72-42f5-9e7f-fe32eee11a0d
# ╟─fb07d4eb-95c0-497a-8fe5-8515026b5b89
# ╠═15bb008a-6b89-4cd6-aad5-5c30da31a7a5
# ╟─75af13de-fcb5-4e1d-804f-dc3cc0809eed
# ╟─21895ece-5cf4-4f6f-b851-44d4fefb1788
# ╟─1ecc4622-b311-47bd-8ba0-d6effa01fa35
# ╠═d12b32d7-4101-44c9-87b1-4ed1438decf7
# ╟─5feec3dd-f53d-4e67-8b1a-ca66131b17eb
# ╟─59d727ac-5610-4443-9783-f1f4b69d863a
# ╟─09f25d8b-7d87-45f7-9a42-68a3027485cf
# ╟─a0ebd9bf-29a6-48c2-ae33-3aa8a25666ca
# ╟─0cd1b4c4-2e56-4027-95b5-24f801f5c04c
# ╟─96b7976d-5e96-4031-8f13-81f1fe14bfc3
# ╠═364199d9-a3f6-4f0f-8f63-60420b9da2c2
