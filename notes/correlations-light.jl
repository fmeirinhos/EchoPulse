### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ ed50b1fe-6a1e-11eb-08ab-95d436eef99c
begin
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
                :(
                    include(mapexpr::Function, x) =
                        $(Expr(:top, :include))(mapexpr, $name, x)
                ),
                :(include($path)),
            ),
        )
        return m
    end
end;

# ╔═╡ 292f7a80-6a1e-11eb-144e-b96c3e89f430
begin
    using Pkg; Pkg.activate()
	
	using LinearAlgebra: diag
	
	using KadanoffBaym, LangrethsRules
	EchoPulse = ingredients("../src/EchoPulse.jl").EchoPulse
	
    using Plots, LaTeXStrings
	Plots.gr()
end;

# ╔═╡ 49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# parameters (units of half-bandwidth)
begin
	# equilibrium parameters
	β = 1.0e3
    ωs_eq = range(-6, 6; length = 2^14 - 1)
	
	# model parameters
    model = EchoPulse.PhotonAssistedSIAM(
        # siam
        ;
        V₀ = sqrt(0.083),# hybridization strength
        ε₀f = -0.4,# d-electron energy

        # quantum light
        Γ₀ = sqrt(0.02),  # light-matter coupling
        ω₀ = 1.2e-2, # photon energy
        Ω₀ = 1.2e-1,

		# other parameters
		η = 0.005, # c-electron width (deactivated)
		γ = 0.03   # impurity fraction
    )

    # time pulse parameters
    Nᵧ = 1.0e0 # maximum photon occupation
    T₀ = 1.0e2 # peak arrival time

    # integration parameters
    tmax = 2.0e2
	rtol = 1e-3
	
	# poor-man's dmft
	dmft = true
	
	# solve with time-matrix initial condition
	timematrix = false
end;

# ╔═╡ a8b39952-53d9-49b6-ada4-d7206640857a
begin
	Gd_eq, Gc_eq, ωK, tdata, t0 = EchoPulse.TimeData(
        model = model,
        ωs = ωs_eq,
		β = β,
        T₀ = T₀,
        Nᵧ = Nᵧ,
		timematrix = timematrix,
		dmft = dmft,
		εs = let
			lb = 0.0001
			ub = 4.0
			
			r = exp.(range(log(lb), log(ub); length=2^7))
			[-reverse(r); [0.0,]; r]
	
			collect(range(-ub, ub; length = 2^9-1))
		end,
	)
end;

# ╔═╡ fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
begin
    cecu6_half_bandwidth = 2.0 #eV
    cdos_half_bandwidth = 3
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth
	
    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 4)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)
	
    md"""
    #### Summary
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the  parameters are
    1. ``\varepsilon_d =`` $(ω_to_eV(model.ε₀f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(model.V₀)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(ωK)) ``ps`` (Kondo temperature & timescale)
    4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
    and the gaussian-shaped pulse parameters were
    5. ``\omega = `` $(ω_to_THz(model.ω₀)) ``THz`` (pulse energy)
    6. ``\Omega = `` $(ω_to_THz(model.Ω₀)) ``THz``  (pulse bandwidth: $(ω_to_ps(model.Ω₀/6)) ``ps``)
    7. ``\Gamma = `` $(ω_to_eV(model.Γ₀)) ``eV`` (light-matter coupling)
    8. ``N_\gamma = `` $(round(Nᵧ, digits=3)) ``n_\gamma`` (number of photons)

    The system will be integrated in the interval (0, $(ω_to_ps(inv(tmax)))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(T₀))) ``ps``.
    """
end

# ╔═╡ 56d36398-6a32-11eb-3f78-01155850c5d2
begin
    plot(
        ωs_eq,
        -imag(Gc_eq);
        label = L"$A_c$",
        linecolor = :darkgreen,
        legend = :topleft,
        xlabel = L"$\omega/D$",
        xlims = 0.75 .* extrema(ωs_eq),
		ylims = (0.0, Inf),
        minorticks = true,
        tick_direction = :out,
        title = L"\textrm{Equilibrium ~spectral ~functions}",
    )

		plot!(ωs_eq, 
		-imag(Gd_eq); 
		label = L"$A_d$", 
		linecolor = :darkorange)

    vline!(
        [model.ε₀f];
        label = L"\varepsilon_d",
        line = (:dot, 1.0, :darkblue, 0.75),
    )

    vline!(
        [-model.ω₀, model.ω₀];
        label = L"\omega_0",
        line = (:dot, 1.0, :darkred, 0.75),
    )

    vspan!(
        [-ωK / 2, ωK / 2];
        fillcolor = :grey,
        alpha = 0.3,
        label = L"\omega_K",
    )
	
	# plot!(ωs_eq,
	# 	π * EchoPulse.gaussian.(ωs_eq, 0.0, 1.0))
	# xlims!(-0.01, 0.01)
end

# ╔═╡ 0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
begin	
	@time ts = kbsolve!(
		(x...) -> EchoPulse.fv!(x..., model, tdata, T₀; pulse=x[2][x[3]] > 50),
		(x...) -> EchoPulse.fd!(x..., model, tdata, T₀; pulse=x[2][x[3]] > 50),
		EchoPulse.initial_condition(tdata),
		(t0, tmax);
		atol = 1e-2 * rtol,
		rtol = rtol,
		dtini = 1e-8,
		γ = 19 / 20,
		callback = (x...) -> begin
			EchoPulse.resize_Σ!(tdata)
			EchoPulse.Σ!(x..., model, tdata)
		end,
		stop = (x...) -> begin
			println(" t: $(round(last(x[1]), digits=3))")
			false
		end
		).t
end;

# ╔═╡ ba97d8ce-2024-4604-ba84-4c2aa80df17c
(; Gd, G0γ, Gγ) = EchoPulse.project(tdata, model; ts=ts, dmft=dmft);

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
    ρtd = (greater(Gd) - lesser(Gd))
	# ρtc = (greater(Ac) - lesser(Ac))
	ρtγ = (greater(G0γ)-lesser(G0γ))
	
	ρτd, (τs, Ts) = wigner_transform_itp(ρtd, ts; 
		fourier = false, 
		# len=ol
	)
    ρωd, (ωs, Ts) = wigner_transform_itp(ρtd, ts; 
		fourier = true, 
		# len=ol
	)
	# ρωc, _ = wigner_transform_itp(ρtc, ts; 
	# 	fourier=true, 
	# 	# len=ol
	# )
	ρτγ = wigner_transform_itp(ρtγ, ts; 
		fourier = false, 
		# len=ol
	)
end;

# ╔═╡ eda788ba-6f84-11eb-24ee-17e261d30bc1
begin
    # Spectral function
    h1 = heatmap(
        Ts,
        ωs,
        -imag(ρωd);
        title = L"A_d(\omega,T)",
        xaxis = nothing,
        ylabel = L"\omega /D",
        minorticks = true,
        tick_direction = :out,
		# ylims = (-1.0, 1.0),
        ylims = (-min(2.0, abs(minimum(ωs))), min(2.0, maximum(ωs))),
        legend = :none,
        c = :magma,
    )

    colorbar = heatmap(
        range(0, 1, length = length(ωs)) .* ones(1, 1),
        legend = nothing,
        xaxis = false,
        yaxis = false,
        label = nothing,
        grid = nothing,
        c = :magma,
        xticks = false,
        yticks = (
            Plots.get_colorbar_ticks(h1.subplots[1])[1] * length(ωs) / maximum(real(ρωd)),
            Plots.get_colorbar_ticks(h1.subplots[1])[2],
        ),
		tick_direction = :out,
        ymirror = true,
    )

    # colorbar = scatter(
    #     [0, 0],
    #     [1, 1];
    #     zcolor = [0, 1],
    #     xaxis = nothing,
    #     xlims = (1, 1),
    #     yaxis = nothing,
    #     label = nothing,
    #     grid = nothing,
    #     c = :magma,
    #     clims = extrema(real(ρωd)),
    # )

    # Occupations
    p1 = plot(
        ts,
        imag(diag(tdata.GfL.data));
        label = L"n_f",
        title = L"\textrm{Field ~occupations}",
        legend = :topleft,
        xlims = extrema(ts),
        minorticks = true,
        tick_direction = :out,
		xlabel = L"T\;D"
        # minorgrid=true
    )
    p1 = plot!(ts, imag(diag(-tdata.GbL.data)); label = L"n_b")
    p1 = plot!(
        ts,
        imag(diag((2tdata.GfL.data - tdata.GbL.data)));
        label = L"Q",
        line = (:dot, :grey, 1),
    )

    twinx_ = twinx()

    # THz pulse
    p1 = plot!(
        twinx_,
        ts,
        G0γ |> (-) ∘ imag ∘ diag ∘ lesser;
        label = L"n^0_\gamma",
        line = (:green, 3.0, 0.2),
        legend = :topright,
        xlims = extrema(ts),
        grid = false,
        minorticks = true,
        tick_direction = :out,
        xaxis = nothing,
    )
	p1 = plot!(
        twinx_,
        ts,
        Gγ |> (-) ∘ imag ∘ diag ∘ lesser;
        label = L"n_\gamma",
        line = (:darkgreen, 1.0),
    )
    # xaxis=nothing)

    # Kondo timescale
	max_pulse_i = findmax(G0γ |> (-) ∘ imag ∘ diag ∘ lesser)[2]
	max_pulse_i = isone(max_pulse_i) ? length(ts) ÷ 2 : max_pulse_i
	
    t_max_pulse = ts[max_pulse_i]
    p1 = vspan!(
        twinx_,
        t_max_pulse .+ 1 / ωK .* [-0.5, +0.5];
        fillcolor = :grey,
        alpha = 0.3,
        linewidth = 0,
        markersize = 0,
        label = L"T_K",
		xaxis = nothing,
    )

    plot(h1, p1, colorbar; layout = @layout [grid(2, 1){0.95w} a])
end

# ╔═╡ 06314e08-579f-4286-81a6-175ca54e5371
plot(Gγ |> lesser |> diag |> imag |> (-) |> diff)

# ╔═╡ dd0b2a88-4eea-4e8c-a669-0099251279d5
begin
	τK = inv(ωK)
	
	Gd_τ = let
        f = EchoPulse.ft(ωs_eq, -1im / π * imag(Gd_eq); mode = -1)
        plot(
            f[1],
            imag(f[2]);
            label = L"\mathcal{F}^{-1}_\omega\{A_d(\omega)\}(\tau)",
            xlabel = L"\tau \;D",
            xlims=τK .* (-2/3, 2/3),
            legend = :topright,
            alpha = 1.0,
            ms = 1.0,
			linecolor = :orange,
        )
    end

    Gd_τ = plot!(
        τs,
        -imag(ρτd[:, max_pulse_i]);
        label = L"A_d(T_0,\tau)",
        ms = 2.0,
		linecolor = :darkgreen
    )
	
	Gd_τ_kondo = let
		# indices of the "Kondo peak"
		w = 10
		i1 = argmin(abs.(ωs_eq .+ w*ωK))
		i2 = argmin(abs.(ωs_eq .- w*ωK))
	
		# Fourier transform to τ-space just the peak
		kondo_ft = EchoPulse.ft(ωs_eq[i1:i2], -1im / π * imag(Gd_eq)[i1:i2]; 
			mode = -1)
	
		Gd_τ_kondo = plot(kondo_ft[1], imag(kondo_ft[2]);
			label = L"\mathcal{F}^{-1}_\omega\{A^{(kondo)}_d(\omega)\}(\tau)",
			xlabel = L"\tau \;D",
			linecolor = :orange,
			xlims=τK .* (-2/3, 2/3)
			)
				
		plot!(Gd_τ_kondo, [], [];
			label = L"A_\gamma(T_0,\tau)",
			linecolor=:darkblue,
		)
		
		plot!(twinx(), τs, -imag(ρτd[:, max_pulse_i]);
			label = L"A_\gamma(T_0,\tau)",
			linecolor=:darkblue,
			grid = false,
			xaxis = nothing,
			legend = nothing,
			xlims=inv(ωK) .* (-2/3, 2/3)
		)
	end
	
	vspan!(Gd_τ_kondo,
        [-τK / 2, τK / 2];
        fillcolor = :grey,
        alpha = 0.3,
        label = L"\tau_K",
    )
	
	plot(Gd_τ, Gd_τ_kondo; layout = grid(2, 1))
    # savefig("equilibrium-comparison.pdf")
end

# ╔═╡ 77a2227a-915c-11eb-00d2-2dbbd3ec938e
# Comparison of frequency *projected* solution with time solution
begin
    Gd_ω = plot(
        ωs_eq,
        -imag(Gd_eq);
        label = L"A_d(\omega) \quad\textrm{(eq)}",
        xlabel = L"\omega / D",
        xlims = (ωs[1], ωs[end]),
        legend = :topright,
        linecolor = :darkorange,
    )

    Gd_ω = plot!(
        ωs,
        -0.5imag(ρωd[:, max_pulse_i]);
        label = L"\mathcal{F}_\tau\{A_d(T_0,\tau)\}(\omega)",
        linecolor = :darkgreen,
    )
	xlims!(-1.0,1.0)
# 	photon_pulse = ((-) ∘ diag)(reverse(greater(G0γ)-lesser(G0γ); dims = 2))
# 	photon_pulse_ωs = EchoPulse.ft(τs, photon_pulse; mode = +1)[2] |> imag
	
# 	Gd_ω = plot!(
# 		twinx(),
# 		ωs,
# 		photon_pulse_ωs;
# 		label = L"A_\gamma(\tau)",
# 		ms = 2.0,
# 		linecolor=:darkred,
# 		xlims = (ωs[1], ωs[end])
# 	)
	
    Gd_ω = vline!(
        [model.ε₀f];
        label = L"\varepsilon_d",
        line = (:dot, 1.0, :darkblue, 0.75),
    )

    Gd_ω = vline!(
        [-model.ω₀, model.ω₀];
        label = L"\omega_0",
        line = (:dot, 1.0, :darkred, 0.75),
    )

	Gd_ω = vspan!(
        [-ωK / 2, ωK / 2];
        fillcolor = :grey,
        alpha = 0.3,
        label = L"\omega_K",
    )
end

# ╔═╡ cfe8984b-3b63-46b9-8f25-ca6c6f6be8ca
begin
	fωd, _ = wigner_transform_itp(im / 2 * (greater(Gd) + lesser(Gd)), ts; 
		fourier=true, 
		# len=ol
	)

	# is this the way to calculate the ratio??
	c_ = let
		Tc1 = model.V₀^2 * Gd
    	Tc2 = im * model.Γ₀^2 * TimeOrderedGF(
	        greater(Gd) .* (greater(G0γ) + transpose(lesser(G0γ))),
	        lesser(Gd) .* (lesser(G0γ) + transpose(greater(G0γ))),
	        ts)
		
		EchoPulse.integrate(tdata.εs, (begin
			G0c_ = TimeOrderedGF(
				exp.(-im * tdata.εs[i] * (ts .- ts')) .* tdata.GcL[i].data, 
				exp.(-im * tdata.εs[i] * (ts .- ts')) .* tdata.GcG[i].data, 
				ts)
			Gc_ = G0c_ + (G0c_ ⋆ (Tc1 + Tc2) ⋆ G0c_)

			fωc, _ = wigner_transform_itp(
				im / 2 * (greater(Gc_) + lesser(Gc_)), ts; 
				fourier=true)
			ρωc, _ = wigner_transform_itp(
				greater(Gc_) - lesser(Gc_), ts; 
				fourier=true)
			
			tdata.dos_c[i] * (fωd ./ ρωd)
		end for i in eachindex(tdata.εs)))

		# i0 = length(tdata.εs) ÷ 2 + 1
		# G0c_ = TimeOrderedGF(
		# 		exp.(-im * tdata.εs[i0] * (ts .- ts')) .* tdata.GcL[i0][1,1], 
		# 		exp.(-im * tdata.εs[i0] * (ts .- ts')) .* tdata.GcG[i0][1,1], 
		# 		ts)
		# Gc_ = G0c_ + (G0c_ ⋆ Tc1 ⋆ G0c_)
		
		# fωc, _ = wigner_transform_itp(
		# 	im / 2 * (greater(Gc_) + lesser(Gc_)), ts; 
		# 	fourier=true)
		# ρωc, _ = wigner_transform_itp(
		# 	greater(Gc_) - lesser(Gc_), ts; 
		# 	fourier=true)

		# (ρωc ./ fωc)
	end
end;

# ╔═╡ 57c01461-f824-430c-846c-775fcb341e1b
begin
	plot(ωs, (1/2 .+ im * (fωd ./ ρωd)[:,max_pulse_i]) |> real;
		label=L"\mathcal{F}_\tau\{f_d(T_0,\tau)\}(\omega)", 
		linecolor=:darkorange
	)
	
	plot!(ωs, (1/2 .+ im * c_[:,max_pulse_i]) |> real;
		label=L"\mathcal{F}_\tau\{f_c(T_0,\tau)\}(\omega)", 
		linecolor=:darkgreen
	)

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
#     function plotter(t)
#         # Spectral function
#         ρt = GᶠL.data .* conj(-GᵇG.data) - GᶠG.data .* conj(-GᵇL.data)

#         ρτ, (τs, Ts) = wigner_transform_itp(ρt, times; fourier = false)
#         ρω, (ωs, Ts) = wigner_transform_itp(ρt, times; fourier = true)

#         h1 = plot(
#             ωs,
#             imag(ρω[:, t]);
#             label = L"A_d",
#             title = L"\textrm{Time spectral function}",
#             xlabel = L"\omega",
#             ylims = extrema(imag(ρω)),
#         )

#         sa(t) = [
#             range(0, 1; length = max(t - 5, 2))
#             ones(max(1, min(5, t)))
#             zeros(length(times) - t)
#         ]

#         # Occupations
#         p1 = plot(
#             ω_to_ps.(inv.(times)),
#             imag ∘ diag(GᶠL.data);
#             label = L"n_f",
#             title = L"\textrm{Field occupations}",
#             seriesalpha = sa(t),
#         )
#         p1 = plot!(
#             ω_to_ps.(inv.(times)),
#             imag ∘ diag(-GᵇL.data);
#             label = L"n_b",
#             seriesalpha = sa(t),
#         )
#         p1 = plot!(
#             ω_to_ps.(inv.(times)),
#             imag ∘ diag((2GᶠL.data - GᵇL.data));
#             label = L"Q",
#             line = (:dot, :grey, 1),
#             legend = :topleft,
#             xlims = ω_to_ps.(inv.(times)),
#             seriesalpha = sa(t),
#         )

#         plt = twinx()

#         p1 = plot!(
#             plt,
#             ω_to_ps.(inv.(times)),
#             (-) ∘ imag ∘ diag(G0L);
#             label = nothing,
#             line = (:green, 2.5),
#             legend = :topright,
#             xlabel = L"T \quad (ps)",#L"T\;D",
#             xlims = ω_to_ps.(inv.(times)),
#             grid = false,
#             seriesalpha = 0.8sa(t),
#         )

#         # 	# Kondo time-scale
#         # 	t_max_pulse = times[findmax(G0L |> (-) ∘ imag ∘ diag)[2]]
#         # 	vspan!(plt, ω_to_ps.(inv.(t_max_pulse .+ 1/ω_data.ωK .* [-0.5, +0.5]));
#         # 		fillcolor=:grey,
#         # 		alpha=0.3,
#         # 		label=L"T_K")

#         return plot(h1, p1; layout = @layout grid(2, 1))
#     end

#     # anim = @animate for t ∈ (length(ω_data.nca.mesh) ÷ 2):length(times)
#     #     plotter(t)
#     # end

#     # gif(anim, "anim_fps15.gif", fps = 15)
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
#     @test all(diag(GᵞG₀.data - GᵞL₀.data) .≈ -1.0im)
#     @test all(imag(diag(GᵞL₀.data)) .<= 0.0)

#     # c-electron checks
#     @test all([GᶜG(t, t) - GᶜL(t, t) for t in ts] .≈ -1.0im)

#     # 	if iszero(model.Γ₀)
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
# ╟─ed50b1fe-6a1e-11eb-08ab-95d436eef99c
# ╠═292f7a80-6a1e-11eb-144e-b96c3e89f430
# ╠═49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# ╠═a8b39952-53d9-49b6-ada4-d7206640857a
# ╟─fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
# ╟─56d36398-6a32-11eb-3f78-01155850c5d2
# ╠═0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
# ╠═ba97d8ce-2024-4604-ba84-4c2aa80df17c
# ╟─a6806902-a209-4b0d-8a2f-cddc1d09677e
# ╟─eda788ba-6f84-11eb-24ee-17e261d30bc1
# ╠═06314e08-579f-4286-81a6-175ca54e5371
# ╟─dd0b2a88-4eea-4e8c-a669-0099251279d5
# ╟─77a2227a-915c-11eb-00d2-2dbbd3ec938e
# ╟─cfe8984b-3b63-46b9-8f25-ca6c6f6be8ca
# ╠═57c01461-f824-430c-846c-775fcb341e1b
# ╟─0483ff48-915c-11eb-12f7-fb5549cf4d66
# ╟─afbf57c4-915f-11eb-02ec-a5e41732d549
# ╟─b7e0f8bc-907a-11eb-21b3-f9471bb006fb
# ╟─9a788356-912b-11eb-0356-35fb2d865d0b
