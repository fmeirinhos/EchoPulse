### A Pluto.jl notebook ###
# v0.19.6

using Markdown
using InteractiveUtils

# ╔═╡ 59060e80-04dd-4b3e-9a24-0df307b9410d
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end;

# ╔═╡ 292f7a80-6a1e-11eb-144e-b96c3e89f430
begin
    using Pkg; Pkg.activate()
	
	using LinearAlgebra: diag
	using KadanoffBaym, LangrethsRules

	const EchoPulse = ingredients("../src/EchoPulse.jl")
	
	using Plots, LaTeXStrings
end; 

# ╔═╡ 658cfe1f-ca33-468f-9c10-6184cfabe7a0
begin
    # Plots.pyplot(); Plots.PyPlot.plt.style.use(["science"])
    
    # Beautiful GR Plots
	Plots.gr(); #Plots.default()
	
    default(
      fontfamily="Computer Modern",
      background_color_legend = nothing,
      foreground_color_legend = nothing,
      tick_direction=:in,
      minorticks=true,
      framestyle = :box,
      size=(600,450),
      titlefontsize = 18, 
      legendfontsize = 12,
      tickfontsize = 12,
      grid=true,
#     dpi=100,
    )

    my_colors = Dict(
      :c => Plots.Colors.colorant"#0C5DA5",
      :d_ω => Plots.Colors.colorant"#00B945",
      :d_t => Plots.Colors.colorant"#FF9500",
      :ωK => Plots.Colors.colorant"#9e9e9e",
      :εd => Plots.Colors.colorant"#845B97",
      :ω0 => Plots.Colors.colorant"#FF2C00"
    )
end;

# ╔═╡ 49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# parameters (units of half-bandwidth)
begin
	# equilibrium parameters
	β = 1.0e3
    ωs_eq = range(-7, 7; length = 2^15 - 1)
	
	# model parameters
    model = EchoPulse.PhotonAssistedSIAM(
        # siam
        ;
        V0 = sqrt(0.08),# hybridization strength
        ε0f = -0.3,# d-electron energy
		U = Inf,

        # quantum light
        Γ0 = sqrt(0.05), # light-matter coupling
        ω0 = 8.0e-2, # photon energy
        Ω0 = 1.5 * 8.0e-2, # photon frequency bandwidth

		# other parameters
		η = 1.0, # c-electron width (deactivated)
		γ = 2e-2   # impurity fraction
    )

    # time pulse parameters
    Nγ = 5.0e-0 # maximum photon occupation
    Tγ = 1.5e2 # peak arrival time

    # integration parameters
    tmax = 3e2
	rtol = 5e-6
	
	# poor-man's dmft
	dmft = false
	
	# solve with time-matrix initial condition
	timematrix = true
end;

# ╔═╡ a8b39952-53d9-49b6-ada4-d7206640857a
begin
	(; GdR_eq, ρcR_eq, ωK, data, t0) = EchoPulse.TimeData(
        model = model,
        ωs = ωs_eq,
		β = β,
        Tγ = Tγ,
        Nγ = Nγ,
		timematrix = timematrix,
		εs = let
			lb = 0.0001
			ub = 4.0
			
			r = exp.(range(log(lb), log(ub); length=2^7))
			[-reverse(r); [0.0,]; r]
	
			dmft ? collect(range(-ub, ub; length = 2^8-1)) : Float64[]
		end,
	)
end;  

# ╔═╡ fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
begin
    cecu6_half_bandwidth = 2.0 # eV
    cdos_half_bandwidth = 3
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth
	
    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 3)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)

	τK = inv(ωK);
	
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

# ╔═╡ 56d36398-6a32-11eb-3f78-01155850c5d2
let
    plot(
        ωs_eq,
        -imag(ρcR_eq);
        label = L"\rho_c",
		linecolor = my_colors[:c],
        xlabel = L"\omega/D",
        title = "Equilibrium spectral functions",
		xlims = 0.75 .* extrema(ωs_eq),
		ylims = (0.0, Inf),
    )

		plot!(ωs_eq, 
		-imag(GdR_eq); 
		label = L"A_d", 
		linecolor = my_colors[:d_ω]
		)

    vline!(
        model.ε0f * [1.0,];
        label = L"\varepsilon_d",
        line = (:dot, 1.5, my_colors[:εd], 0.75),
    )

    vline!(
        model.ω0 * [-1.0, 1.0];
        label = L"\omega_0",
        line = (:dot, 1.5, my_colors[:ω0], 0.75),
    )

    vspan!(
        ωK * [-1/2, 1/2];
        fillcolor = my_colors[:ωK],
        alpha = 0.3,
        label = L"\omega_K",
		linewidth = 0,
        markersize = 0,
    )
end

# ╔═╡ 0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
begin
	# Time-step
	@time ts = kbsolve!(
		(x...) -> EchoPulse.fv!(x..., model, data; Tγ=Tγ, pulse=true),
		(x...) -> EchoPulse.fd!(x..., model, data; Tγ=Tγ, pulse=true),
		EchoPulse.initial_condition(data),
		(collect(t0), tmax);
		rtol = rtol,
		atol = 1e-3 * rtol,
		callback = (x...) -> begin
			EchoPulse.resize_Σ!(data)
			EchoPulse.Σ!(x..., model, data)
		end,
		stop = (x...) -> begin
			println(" t: $(round(last(x[1]), digits=3))")
			false
		end
		).t

	# Post processing
	EchoPulse.resize_Σ!(data; resize=true)

	# Calculate observables
	(; Gd, ρc, G0γ, Gγ, Gγ_out, Gγ_out_vac) = EchoPulse.project(data, model; ts=ts, dmft=dmft);
end;

# ╔═╡ 32062fb9-70ed-4c93-a216-78c2628b963e
# heatmap(ts, ts, data.TcG.data |> imag)

# ╔═╡ a6806902-a209-4b0d-8a2f-cddc1d09677e
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
	ρc_t = (greater(ρc) - lesser(ρc))
	ργ_t = (greater(G0γ)-lesser(G0γ))
	
	ρd_τ, (τs, Ts) = wigner_transform_itp(ρd_t, ts; 
		fourier = false, 
		# len=ol
	)
    ρd_ω, (ωs, Ts) = wigner_transform_itp(ρd_t, ts; 
		fourier = true, 
		# len=ol
	)
	ρc_ω, _ = wigner_transform_itp(ρc_t, ts; 
		fourier=true, 
		# len=ol
	)
	ργ_τ, _ = wigner_transform_itp(ργ_t, ts; 
		fourier = false, 
		# len=ol
	)
end;

# ╔═╡ e8cee7c6-39e5-49f3-871a-88c9c705ce04
let
	plot(ts, -imag(diag(lesser(G0γ))), 
		label=L"n_\gamma^\mathrm{in}", 
		xlabel = L"T\;D",
		line=(:darkgreen),
		ylims=(0.0, Nγ),
		# legend=:right,
		left_margin = 5Plots.mm,
		right_margin = 15Plots.mm,
		title="Photon output"
	)
	
	plot!([], [], 
		label=L"n_\gamma^\mathrm{out} - n_\gamma^\mathrm{in}", 
		color=:orange
	)

	vspan!(Tγ .+ τK .* [-0.5, +0.5];
		label = L"\tau_K",
		fillcolor = :grey,
		alpha = 0.1,
		linewidth = 0,
		markersize = 0,
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

	Gγ = -imag(diag(lesser(Gγ_out_vac)))
	plot!(twinx(), ts[i0:end], Gγ[i0:end];
		color=:orange,
		label=nothing, ymirror = true,
		# ylims=(Gγ[end], maximum(Gγ))
	)
	# plot!(twinx(), ts, Gγ;
	# 	color=:orange,
	# 	label=nothing, ymirror = true,
	# 	# ylims=(Gγ[end], maximum(Gγ))
	# )
	
	xlims!(0.0, ts[end])
	# ylims!(0.0, 0.1)
	# savefig("test.pdf")
end

# ╔═╡ eda788ba-6f84-11eb-24ee-17e261d30bc1
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
		# ylims = (-1.0, 1.0),
        ylims = (-min(2.0, abs(minimum(ωs))), min(2.0, maximum(ωs))),
        legend = :none,
        c = :magma,
		rightmargin=5Plots.mm
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
        yticks = (
            Plots.get_colorbar_ticks(h1.subplots[1])[1] * 
			length(ωs) / h1.subplots[1][1][:clims_calculated][2],
            Plots.get_colorbar_ticks(h1.subplots[1])[2],
        ),
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
		rightmargin=5Plots.mm,
        # minorgrid=true
    )
    p1 = plot!(ts, imag(diag(-data.GbL.data)); 
		label = L"n_b")
	
    p1 = plot!(
        ts,
        imag(diag((2data.GfL.data - data.GbL.data - data.GaL.data)));
        label = L"Q",
        line = (:dot, :grey, 1),
    )

	# ylims!(0.11, 0.12)
	# xlims!(-10,500)
	# # Kondo timescale
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

    plot(h1, p1, colorbar;
		size=(600,450) .* 1.3,
		layout = @layout [grid(2, 1){0.95w} a]
	)
end

# ╔═╡ 77a2227a-915c-11eb-00d2-2dbbd3ec938e
# Comparison of frequency *projected* solution with time solution
let
    plot(
        ωs_eq,
        -2imag(GdR_eq);
        label = L"A_d(\omega) \quad\mathrm{(eq)}",
        xlabel = L"\omega / D",
        xlims = (ωs[1], ωs[end]),
        legend = :topright,
        linecolor = my_colors[:d_ω],
		title = "Kondo peak melt"
    )

	max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]
		
    plot!(
        ωs,
        -imag(ρd_ω[:, max_pulse_i]);
        label = L"\mathcal{F}_\tau\{A_d(T_0,\tau)\}(\omega)",
        linecolor = my_colors[:d_t],
    )
	
    vline!(
        [model.ε0f];
        label = L"\varepsilon_d",
        line = (:dot, 1.5, my_colors[:εd], 0.75),
    )

    vline!(
        [-model.ω0, model.ω0];
        label = L"\omega_0",
        line = (:dot, 1.5, my_colors[:ω0], 0.75),
    )

	vspan!(
        ωK * [-1/2, 1/2];
        fillcolor = my_colors[:ωK],
        alpha = 0.3,
        label = L"\omega_K",
		linewidth = 0,
        markersize = 0,
    )

	xlims!(-1.0,1.0)
end

# ╔═╡ dd0b2a88-4eea-4e8c-a669-0099251279d5
let
	Gd_τ = let
        f = EchoPulse.ft(ωs_eq, -2im * imag(GdR_eq); inverse = true)
        plot(
            f[1],
            imag(f[2]);
            label = L"\mathcal{F}^{-1}_\omega\{A_d(\omega)\}(\tau)",
            xlabel = L"\tau \;D",
            xlims=τK .* (-2/3, 2/3),
            legend = :topright,
            alpha = 1.0,
            ms = 1.0,
			linecolor = my_colors[:d_ω],
        )
    end

	max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]
	
    Gd_τ = plot!(
        τs,
        -imag(ρd_τ[:, max_pulse_i]);
        label = L"A_d(T_0,\tau)",
        ms = 2.0,
		linecolor = my_colors[:d_t]
    )
	
	Gd_τ_kondo = let
		# indices of the "Kondo peak"
		w = 10
		i1 = argmin(abs.(ωs_eq .+ w*ωK))
		i2 = argmin(abs.(ωs_eq .- w*ωK))
	
		# Fourier transform to τ-space just the peak
		kondo_ft = EchoPulse.ft(ωs_eq[i1:i2], -2im * imag(GdR_eq)[i1:i2]; 
			inverse = true)
	
		Gd_τ_kondo = plot(kondo_ft[1], imag(kondo_ft[2]);
			label = L"\mathcal{F}^{-1}_\omega\{A^{(kondo)}_d(\omega)\}(\tau)",
			xlabel = L"\tau \;D",
			linecolor = my_colors[:d_ω],
			xlims=τK .* (-2/3, 2/3)
			)
				
		plot!(Gd_τ_kondo, [], [];
			label = L"A_\gamma(T_0,\tau)",
			linecolor = my_colors[:ω0],
		)
		
		plot!(twinx(), τs, -imag(ργ_τ[:, max_pulse_i]);
			label = L"A_\gamma(T_0,\tau)",
			linecolor= my_colors[:ω0],
			xaxis = nothing,
			legend = nothing,
			xlims=inv(ωK) .* (-2/3, 2/3)
		)
	end

	vspan!(
		Gd_τ_kondo,
        τK * [-1/2, 1/2];
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

# ╔═╡ cfe8984b-3b63-46b9-8f25-ca6c6f6be8ca
begin
	fd_ω, _ = wigner_transform_itp(im / 2 * (greater(Gd) + lesser(Gd)), ts; 
		fourier=true, 
		# len=ol
	)

	# is this the way to calculate the ratio??
	# c_ = let
	# 	Tc1 = model.V0^2 * Gd
 #    	Tc2 = im * model.Γ0^2 * TimeOrderedGF(
	#         greater(Gd) .* (greater(G0γ) + transpose(lesser(G0γ))),
	#         lesser(Gd) .* (lesser(G0γ) + transpose(greater(G0γ))),
	#         ts)
		
	# 	EchoPulse.integrate(tdata.εs, (begin
	# 		G0c_ = TimeOrderedGF(
	# 			exp.(-im * tdata.εs[i] * (ts .- ts')) .* tdata.GcL[i].data, 
	# 			exp.(-im * tdata.εs[i] * (ts .- ts')) .* tdata.GcG[i].data, 
	# 			ts)
	# 		Gc_ = G0c_ + (G0c_ ⋆ (Tc1 + Tc2) ⋆ G0c_)

	# 		fωc, _ = wigner_transform_itp(
	# 			im / 2 * (greater(Gc_) + lesser(Gc_)), ts; 
	# 			fourier=true)
	# 		ρωc, _ = wigner_transform_itp(
	# 			greater(Gc_) - lesser(Gc_), ts; 
	# 			fourier=true)
			
	# 		tdata.dos_c[i] * (fωd ./ ρωd)
	# 	end for i in eachindex(tdata.εs)))

	# 	# i0 = length(tdata.εs) ÷ 2 + 1
	# 	# G0c_ = TimeOrderedGF(
	# 	# 		exp.(-im * tdata.εs[i0] * (ts .- ts')) .* tdata.GcL[i0][1,1], 
	# 	# 		exp.(-im * tdata.εs[i0] * (ts .- ts')) .* tdata.GcG[i0][1,1], 
	# 	# 		ts)
	# 	# Gc_ = G0c_ + (G0c_ ⋆ Tc1 ⋆ G0c_)
		
	# 	# fωc, _ = wigner_transform_itp(
	# 	# 	im / 2 * (greater(Gc_) + lesser(Gc_)), ts; 
	# 	# 	fourier=true)
	# 	# ρωc, _ = wigner_transform_itp(
	# 	# 	greater(Gc_) - lesser(Gc_), ts; 
	# 	# 	fourier=true)

	# 	# (ρωc ./ fωc)
	# end
end;

# ╔═╡ 57c01461-f824-430c-846c-775fcb341e1b
let
	max_pulse_i = findmin(x -> abs(x - Tγ), ts)[2]
	
	plot(ωs, (1/2 .+ im * (fd_ω ./ ρd_ω)[:,max_pulse_i]) |> real;
		label = L"\mathcal{F}_\tau\{f_d(T_0,\tau)\}(\omega)", 
		linecolor = my_colors[:d_t]
	)
	
	# plot!(ωs, (1/2 .+ im * c_[:,max_pulse_i]) |> real;
	# 	label=L"\mathcal{F}_\tau\{f_c(T_0,\tau)\}(\omega)", 
	# 	linecolor=:darkgreen
	# )

	plot!(ωs, 1.0 ./ (exp.(β .* ωs) .+ 1); 
    		label=L"f_{eq}(\omega)", 
    		xlabel=L"\omega / D",
			xlims=(ωs[1], ωs[end]),
    		legend=:topright,
    		line=(:dot, :grey)
	)
	
	ylims!(-0.1, 1.1)
	xlims!(-0.1,0.1)	
end

# ╔═╡ 0483ff48-915c-11eb-12f7-fb5549cf4d66
# begin
#     Q = EchoPulse.integrate(nca.mesh.mesh, -imG[:, 1, 2] + 2imG[:, 2, 2]) / (2π)
#     ts_ft = EchoPulse.ft(nca.mesh.mesh, zero(nca.mesh.mesh); mode = -1)[1]

#     # Lesser
#     L = mapslices(
#         x -> EchoPulse.ft(nca.mesh.mesh, x / (2pi); mode = -1)[2],
#         1.0im * hcat(
#             imG[:, :, 2] / Q,
#             +nca.data.fermi .* nca.data.ρc,
#             -nca.data.bose .* nca.data.ργ,
#         );
#         dims = 1,
#     )

#     # Greater
#     G = mapslices(
#         x -> EchoPulse.ft(nca.mesh.mesh, x / (2pi); mode = -1)[2],
#         1.0im * hcat(
#             imG[:, :, 1],
#             -nca.data.fermi⁻ .* nca.data.ρc,
#             -nca.data.bose⁻ .* nca.data.ργ,
#         );
#         dims = 1,
#     )

#     xlims = 0.25 .* extrema(τs)
#     xlabel = L"\tau D"

# 	GbLτ = plot(
# 		ts_ft,
# 		imag(L[:, 1]);
# 		label = L"\mathcal{F}_\omega\{G^<_b\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GbLτ = plot!(τs, imag(diag(reverse(tdata.GbL.data; dims = 2))); label = L"G^<_b(\tau)")

# 	GbGτ = plot(
# 		ts_ft,
# 		imag(G[:, 1]);
# 		label = L"\mathcal{F}_\omega\{G^>_b\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GbGτ = plot!(τs, imag(diag(reverse(tdata.GbG.data; dims = 2))); label = L"G^>_b(\tau)")

# 	GfLτ = plot(
# 		ts_ft,
# 		imag(L[:, 2]);
# 		label = L"\mathcal{F}_\omega\{G^<_f\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GfLτ = plot!(τs, imag(diag(reverse(tdata.GfL.data; dims = 2))); label = L"G^<_f(\tau)")

# 	GfGτ = plot(
# 		ts_ft,
# 		imag(G[:, 2]);
# 		label = L"\mathcal{F}_\omega\{G^>_f\}(\tau)",
# 		legend = :topright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GfGτ = plot!(τs, imag(diag(reverse(tdata.GfG.data; dims = 2))); label = L"G^>_f(\tau)")

# 	GcLτ = plot(
# 		ts_ft,
# 		imag(L[:, 3]);
# 		label = L"\mathcal{F}_\omega\{G^<_c\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GcLτ = plot!(
# 		τs,
# 		imag(diag(reverse(lesser(G0c); dims = 2)));
# 		label = L"G^<_c(\tau)",
# 	)

# 	GcGτ = plot(
# 		ts_ft,
# 		imag(G[:, 3]);
# 		label = L"\mathcal{F}_\omega\{G^>_c\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GcGτ = plot!(
# 		τs,
# 		imag(diag(reverse(greater(G0c); dims = 2)));
# 		label = L"G^>_c(\tau)",
# 	)

# 	GγLτ = plot(
# 		ts_ft,
# 		imag(L[:, 4]);
# 		label = L"\mathcal{F}_\omega\{G^<_\gamma\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GγLτ = plot!(τs, imag(diag(reverse(lesser(G0γ); dims = 2))); label = L"G^<_\gamma(\tau)")

# 	GγGτ = plot(
# 		ts_ft,
# 		imag(G[:, 4]);
# 		label = L"\mathcal{F}_\omega\{G^>_\gamma\}(\tau)",
# 		legend = :bottomright,
# 		xlims = xlims,
# 		xlabel = xlabel,
# 	)
# 	GγGτ = plot!(τs, imag(diag(reverse(greater(G0γ); dims = 2))); label = L"G^>_\gamma(\tau)")

# 	plot(GbLτ, GbGτ, GfLτ, GfGτ, GcLτ, GcGτ, GγLτ, GγGτ; layout = @layout [a b; c d; e f; g h])
# end

# ╔═╡ afbf57c4-915f-11eb-02ec-a5e41732d549
# begin
#     imΣ = EchoPulse.NCA.slave_particle_imΣ(nca, imG)

#     cap(x) = x.data[1:length(ts), 1:length(ts)]

#     ΣL = mapslices(
#         x -> EchoPulse.ft(nca.mesh.mesh, im / 2pi * x; mode = -1)[2],
#         imΣ[:, :, 2] / Q;
#         dims = 1,
#     )
#     ΣG = mapslices(x -> EchoPulse.ft(nca.mesh.mesh, im / 2pi * x; mode = -1)[2], imΣ[:, :, 1]; dims = 1)

#     ΣᵇLτ = plot(
#         ts_ft,
#         imag(ΣL[:, 1]);
#         label = L"\mathcal{F}_\omega\{\Sigma^<_b\}(\tau)",
#         legend = :bottomright,
#         xlims = xlims,
#         xlabel = xlabel,
#     )
#     ΣᵇLτ =
#         plot!(τs, imag(diag(reverse(cap(tdata.ΣbL); dims = 2))); label = L"\Sigma^<_b(\tau)")

#     ΣᵇGτ = plot(
#         ts_ft,
#         imag(ΣG[:, 1]);
#         label = L"\mathcal{F}_\omega\{\Sigma^>_b\}(\tau)",
#         legend = :bottomright,
#         xlims = xlims,
#         xlabel = xlabel,
#     )
#     ΣᵇGτ =
#         plot!(τs, imag(diag(reverse(cap(tdata.ΣbG); dims = 2))); label = L"\Sigma^>_b(\tau)")

#     ΣᶠLτ = plot(
#         ts_ft,
#         imag(ΣL[:, 2]);
#         label = L"\mathcal{F}_\omega\{\Sigma^<_f\}(\tau)",
#         legend = :topright,
#         xlims = xlims,
#         xlabel = xlabel,
#     )
#     ΣᶠLτ =
#         plot!(τs, imag(diag(reverse(cap(tdata.ΣfL); dims = 2))); label = L"\Sigma^<_f(\tau)")

#     ΣᶠGτ = plot(
#         ts_ft,
#         imag(ΣG[:, 2]);
#         label = L"\mathcal{F}_\omega\{\Sigma^>_f\}(\tau)",
#         legend = :bottomright,
#         xlims = xlims,
#         xlabel = xlabel,
#     )
#     ΣᶠGτ =
#         plot!(τs, imag(diag(reverse(cap(tdata.ΣfG); dims = 2))); label = L"\Sigma^>_f(\tau)")

#     plot(ΣᵇLτ, ΣᵇGτ, ΣᶠLτ, ΣᶠGτ; layout = @layout [a b; c d])
# end

# ╔═╡ b7e0f8bc-907a-11eb-21b3-f9471bb006fb
# begin
# 	gl0 = [-imag(tdata.GγL0_f(t,t)) for t in ts]
# 	xlims_ = (0.0, ts[1197+1200])
# 	ylims_ = extrema(gl0)
# 	ylims__ = extrema(-0.5imag(ρωd[:,1197+200]))
# 	Gd_eq__ = -0.5imag(ρωd[:,1197+200])
# 	#extrema(-0.5imag(ρωd))
	
#     function plotter(t)
# 		h1 = plot(
# 	        ωs,
# 	        Gd_eq__;
# 	        label = L"A^{\textrm{(eq)}}_d(\omega)",
# 	        xlabel = L"\omega / D",
# 	        xlims = (ωs[1], ωs[end]),
# 	        legend = :topright,
# 	        linecolor = :darkorange,
# 	    )
# 		# h1 = plot(
# 	 #        ωs_eq,
# 	 #        -imag(Gd_eq);
# 	 #        label = L"A^{\textrm{(eq)}}_d(\omega)",
# 	 #        xlabel = L"\omega / D",
# 	 #        xlims = (ωs[1], ωs[end]),
# 	 #        legend = :topright,
# 	 #        linecolor = :darkorange,
# 	 #    )
	
# 	    h1 = plot!(
# 	        ωs,
# 	        -0.5imag(ρωd[:, t]);
# 	        label = L"A_d(t,\omega)",
# 	        linecolor = :darkblue,
# 			ylims = ylims__, margin=0mm
# 	    )

#         # sa(t) = [
#         #     range(0, 1; length = max(t - 5, 2))
#         #     ones(max(1, min(5, t)))
#         #     zeros(length(ts) - t)
#         # ]
# 		# sa(t) = range(0, 1; length=t)
# 		# sa(t) = let
# 		# 	v = max(1, min(5,t))
# 		# 	[
# 		# 		t-v == 1 ? [0.0] : range(0, 1; length=max(t-v, 0))
# 		# 		ones(v)
# 		# 	]
# 		# end
# 		sa(t) = let
# 			v = max(1, min(5,t))
# 			[
# 				1 # needed to get the correct label marker
# 				t-v == 1 ? [0.0] : range(0, 1; length=max(t-v, 0))
# 				ones(v-1)
# 			]
# 		end

# 		f_ = [+imag(tdata.GfL.data[i,i]) for i in 1:t]
# 		b_ = [+imag(-tdata.GbL.data[i,i]) for i in 1:t]

#         # Occupations
# 		p1 = plot(
#             ts[1:t],
#             2f_ + b_;
#             label = L"Q",
#             line = (:dot, :grey, 1),
#             legend = :topright,
#             xlims = xlims_,
# 			ylims = (0.0, 1.0),
#             linealpha = sa(t),
# 			# margin=5mm
#         )
		
#         p1 = plot!(
#             ts[1:t],
#             f_;
#             label = L"n_f",
# 			color = :darkblue,
#             # title = L"\textrm{Field occupations}",
#             linealpha = sa(t),
#         )
#         p1 = plot!(
#             ts[1:t],
#             b_;
#             label = L"n_b",
# 			color = :darkorange,
#             linealpha = sa(t),
#         )
		
# 		p1 = plot!(
#             ts[1:1],
#             gl0[1:1];
# 			label = L"n_\gamma",
#             line = (:darkgreen, 2.5),
#             grid = false,
#             seriesalpha = 0.8sa(t),
#         )
		
# 		# Kondo time-scale
# 	    p1 = vspan!(
# 	        p1,
# 	        T0 .+ 1 / ωK .* [-0.5, +0.5];
# 	        fillcolor = :grey,
# 	        alpha = 0.1,
# 	        linewidth = 0,
# 	        markersize = 0,
# 	        label = L"T_K",
# 	    )
		
#         plt = twinx()
		
# 		p1 = plot!(
#             plt,
#             ts[1:t],
#             gl0[1:t];
#             label = nothing,
#             line = (:green, 3.5),
#             legend = :topright,
#             xlabel = L"T\;D",
#             xlims = xlims_,
# 			ylims = (0.0, 10.),
#             grid = false,
#             seriesalpha = 0.8sa(t),
# 			margin=5mm
#         )

#         return plot(h1, p1;
# 			# size=2//3 .* (1920, 1080),
# 			size=1//3 .* (1920, 1080), 
# 			layout = @layout grid(2, 1)
# 		)
#     end

#     anim = @animate for t in 1197 .+ range(0, 1200, step=4)
# 		iszero(mod(t, 10)) ? (@info ts[t]) : nothing
#         plotter(t)
#     end

#     # gif(anim, "anim_fps15.mp4", fps = 24)
# end

# ╔═╡ 9a788356-912b-11eb-0356-35fb2d865d0b
# # Kubo-Martin-Schwinger boundary conditions
# @testset "KMS boundary conditions" begin
#     # pseudo-particles checks
#     @test all(diag(GᶠG.data) .≈ -1.0im)
#     @test all(diag(GᵇG.data) .≈ -1.0im)

#     @test maximum(+imag(GᶠL.data)) <= 1.0 # Maximum occupation is 1
#     @test maximum(-imag(GᵇL.data)) <= 1.0 # Maximum occupation is 1
#     @test maximum(-imag(GᶠG.data)) <= 1.0
#     @test maximum(-imag(GᵇG.data)) <= 1.0

#     @test minimum(+imag(GᶠL.data)) >= -1.0
#     @test minimum(-imag(GᵇL.data)) >= -1.0
#     @test minimum(-imag(GᶠG.data)) >= -1.0
#     @test minimum(-imag(GᵇG.data)) >= -1.0

#     @test all(imag(diag(-GᵇL.data + 2GᶠL.data)) .≈ 1.0)

#     # photon checks
#     @test all(diag(GᵞG0.data - GᵞL0.data) .≈ -1.0im)
#     @test all(imag(diag(GᵞL0.data)) .<= 0.0)

#     # c-electron checks
#     @test all([GᶜG(t, t) - GᶜL(t, t) for t in ts] .≈ -1.0im)

#     # 	if iszero(model.Γ0)
#     # 		∫dω(f) = M.integrate(freqs.mesh, f, M.TrapezoidalEvenFast()) / (2π)
#     # 		≅(a,b) = isapprox(a,b;atol=5e-3)

#     # 		# C^<(t,t0) = C˺(t,t0)
#     # 		@test all(GᶜL[t,1] ≅ +∫dω(fermi .* (-conj(Aᶜ˹[t]))) for t in 1:size(Aᶜ˹,2))
#     # 		@test all(GᶠL[t,1] ≅ +∫dω(fermi .* (-conj(GᶠT[t]))) for t in 1:size(GᶠT,2))
#     # 		@test all(GᵇL[t,1] ≅ -∫dω(fermi .* (-conj(GᵇT[t]))) for t in 1:size(GᶠT,2))
#     # 		@test all(ΣᶠL[t,1] ≅ +∫dω(fermi .* (-conj(ΣᶠT[t]))) for t in 1:size(ΣᶠT,2))
#     # 		@test all(ΣᵇL[t,1] ≅ -∫dω(fermi .* (-conj(ΣᵇT[t]))) for t in 1:size(ΣᵇT,2))

#     # 		# C^>(t0,t) = C˹(t0,t)
#     # 		@test all(GᶜG[1,t] ≅ -∫dω(fermi⁻ .* Aᶜ˹[t]) for t in 1:size(Aᶜ˹,2))
#     # 		@test all(GᶠG[1,t] ≅ -∫dω(fermi⁻ .* GᶠT[t]) for t in 1:size(GᶠT,2))
#     # 		@test all(GᵇG[1,t] ≅ -∫dω(fermi⁻ .* GᵇT[t]) for t in 1:size(GᵇT,2))
#     # 		@test all(ΣᶠG[1,t] ≅ -∫dω(fermi⁻ .* ΣᶠT[t]) for t in 1:size(ΣᶠT,2))
#     # 		@test all(ΣᵇG[1,t] ≅ -∫dω(fermi⁻ .* ΣᵇT[t]) for t in 1:size(ΣᵇT,2))
#     # 		else
#     # 		println("Cannot test mixed functions BCs")
#     # 	end
# end;

# ╔═╡ Cell order:
# ╟─59060e80-04dd-4b3e-9a24-0df307b9410d
# ╠═292f7a80-6a1e-11eb-144e-b96c3e89f430
# ╟─658cfe1f-ca33-468f-9c10-6184cfabe7a0
# ╠═49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# ╠═a8b39952-53d9-49b6-ada4-d7206640857a
# ╟─fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
# ╟─56d36398-6a32-11eb-3f78-01155850c5d2
# ╠═0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
# ╠═32062fb9-70ed-4c93-a216-78c2628b963e
# ╟─a6806902-a209-4b0d-8a2f-cddc1d09677e
# ╠═e8cee7c6-39e5-49f3-871a-88c9c705ce04
# ╠═eda788ba-6f84-11eb-24ee-17e261d30bc1
# ╟─77a2227a-915c-11eb-00d2-2dbbd3ec938e
# ╟─dd0b2a88-4eea-4e8c-a669-0099251279d5
# ╟─cfe8984b-3b63-46b9-8f25-ca6c6f6be8ca
# ╟─57c01461-f824-430c-846c-775fcb341e1b
# ╟─0483ff48-915c-11eb-12f7-fb5549cf4d66
# ╟─afbf57c4-915f-11eb-02ec-a5e41732d549
# ╟─b7e0f8bc-907a-11eb-21b3-f9471bb006fb
# ╟─9a788356-912b-11eb-0356-35fb2d865d0b
