### A Pluto.jl notebook ###
# v0.16.0

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
    using Revise
	
	EchoPulse = ingredients("../src/EchoPulse.jl").EchoPulse
	
    using KadanoffBaym, LangrethsRules

    using LinearAlgebra: diag
    using UnPack

    using Test

    using Plots, LaTeXStrings
	Plots.gr()
end;

# ╔═╡ 49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# parameters (units of half-bandwidth)
begin
	# equilibrium parameters
	β = 4.0e3
    ωmesh = range(-8, 8; length = 2^15)
	
	# model parameters
    model = EchoPulse.PhotonAssistedSIAM(
        # siam
        ;
        V₀ = sqrt(0.080),# hybridization strength
        ε₀f = -0.4,# d-electron energy

        # quantum light
        Γ₀ = sqrt(0.03),  # light-matter coupling
        ω₀ = 1.0e-1, # photon energy
        Ω₀ = 1.0e-0,
    )

    # time pulse parameters
    Nᵧ = 1.0e1 # maximum photon occupation
    T₀ = 5.0e1 # peak arrival time

    # integration parameters
    tmax = 1.0e2
	rtol = 1e-3    
end;

# ╔═╡ a8b39952-53d9-49b6-ada4-d7206640857a
begin
	nca, imG, Geq, ωK, tdata, t0 = EchoPulse.TimeData(
        model = model,
        mesh = ωmesh,
		β = β,
        T₀ = T₀,
        Nᵧ = Nᵧ,
        timematrix = false,
		# ωᶜ = collect(range(-2.0, 2.0; length = 2^9))
		ωᶜ = nothing
	)
end;

# ╔═╡ fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
begin
    cecu6_half_bandwidth = 2.0
    cdos_half_bandwidth = 3
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth
    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 4)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)
    md"""
    #### Summary
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the chosen SIAM parameters were
    1. ``\varepsilon_d =`` $(ω_to_eV(model.ε₀f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(model.V₀)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(ωK)) ``K`` (Kondo temperature: $(ω_to_ps(ωK)) ``ps``)
    4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
    and the gaussian-shaped pulse parameters were
    5. ``\omega = `` $(ω_to_THz(model.ω₀)) ``THz`` (pulse energy)
    6. ``\Omega = `` $(ω_to_THz(model.Ω₀)) ``THz``  (pulse bandwidth: $(ω_to_ps(model.Ω₀)) ``ps``)
    7. ``\Gamma = `` $(ω_to_eV(model.Γ₀)) ``eV`` (light-matter coupling)
    8. ``N_\gamma = `` $(round(Nᵧ, digits=3)) ``n_\gamma`` (number of photons)

    The system will be integrated for to $(ω_to_ps(1/tmax)) ps with the light-pulse reaching its maximum at $(ω_to_ps(1/T₀)) ps.
    """
end

# ╔═╡ 56d36398-6a32-11eb-3f78-01155850c5d2
begin
    ωs_plots = ωmesh # .|> ω_to_eV

    plot(
        ωs_plots,
        imag(Geq[1]);
        label = L"$A_c$",
        linecolor = :green,
        legend = :topleft,
        xlabel = L"$\omega/D$",
        xlims = 0.75 .* extrema(ωs_plots),
        minorticks = true,
        tick_direction = :out,
        title = L"\textrm{Equilibrium ~spectral ~functions}",
    )

    plot!(ωs_plots, 
		imag(Geq[2]); 
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
end

# ╔═╡ 0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
begin
	@time ts = kbsolve!(
		(u, ts, t1, t2) -> EchoPulse.fv!(u, model, tdata, ts, t1, t2),
		(u, ts, t1, t2) -> EchoPulse.fd!(u, model, tdata, ts, t1, t2),
		EchoPulse.initial_condition(tdata),
		(t0, tmax);
		atol = 1e-2 * rtol,
		rtol = rtol,
		dtini = 1e-10,
		γ = 9 / 10,
		callback = (ts, t1, t2) -> EchoPulse.Σ!(model, tdata, ts, t1, t2),
		stop = (_, ts) -> (println(" t: $(last(ts))"); false)
		).t
end;

# ╔═╡ 06020ea7-d32d-4f06-972b-a711aab1631b
begin
    Gd = TimeOrderedGF(
        1.0im * tdata.GfL.data .* conj(-tdata.GbG.data),
        1.0im * tdata.GfG.data .* conj(-tdata.GbL.data),
        ts,
    )
    G0γ = TimeOrderedGF(
        [tdata.GγL0(t1, t2) for t1 in ts, t2 in ts],
        [tdata.GγG0(t1, t2) for t1 in ts, t2 in ts],
        ts,
    )

    if isnothing(tdata.ωc)
        G0c = TimeOrderedGF(
            [tdata.GcL(t1, t2) for t1 in ts, t2 in ts],
            [tdata.GcG(t1, t2) for t1 in ts, t2 in ts],
            ts,
        )
        Gc = G0c + (G0c ⋆ (model.V₀^2 * Gd) ⋆ G0c)
        Tγ = TimeOrderedGF(
            model.Γ₀^2 *
            1.0im *
            (lesser(Gc) .* conj(greater(Gd)) + lesser(Gd) .* conj(greater(Gc))),
            model.Γ₀^2 *
            1.0im *
            (greater(Gc) .* conj(lesser(Gd)) + greater(Gd) .* conj(lesser(Gc))),
            ts,
        )

    else
        G0c = [
            TimeOrderedGF(tdata.GcL[i].data, tdata.GcG[i].data, ts) for
            i = 1:length(tdata.GcG)
        ]
        Gc = [G0c[i] + (G0c[i] ⋆ (model.V₀^2 * Gd) ⋆ G0c[i]) for i = 1:length(tdata.GcG)]
        Tγ = sum(
            TimeOrderedGF(
                model.Γ₀^2 *
                tdata.ρc[i] *
                1.0im *
                (lesser(Gc[i]) .* conj(greater(Gd)) + lesser(Gd) .* conj(greater(Gc[i]))),
                model.Γ₀^2 *
                tdata.ρc[i] *
                1.0im *
                (greater(Gc[i]) .* conj(lesser(Gd)) + greater(Gd) .* conj(lesser(Gc[i]))),
                ts,
            ) for i = 1:length(tdata.GcG)
        )
    end
end;

# ╔═╡ eda788ba-6f84-11eb-24ee-17e261d30bc1
begin
    # Spectral function
    ρt = greater(Gd) - lesser(Gd)

    ρτ, (τs, Ts) = wigner_transform_itp(ρt, ts; fourier = false)
    ρω, (ωs, Ts) = wigner_transform_itp(ρt, ts; fourier = true)

    ts_plots = ts

    # Spectral function
    h1 = heatmap(
        Ts,
        ωs,
        real(ρω);
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
            Plots.get_colorbar_ticks(h1.subplots[1])[1] * length(ωs) / maximum(real(ρω)),
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
    #     clims = extrema(real(ρω)),
    # )

    # Occupations
    p1 = plot(
        ts_plots,
        imag(diag(tdata.GfL.data));
        label = L"n_f",
        title = L"\textrm{Field ~occupations}",
        legend = :topleft,
        xlims = extrema(ts_plots),
        minorticks = true,
        tick_direction = :out,
		xlabel = L"T\;D"
        # minorgrid=true
    )
    p1 = plot!(ts_plots, imag(diag(-tdata.GbL.data)); label = L"n_b")
    p1 = plot!(
        ts_plots,
        imag(diag((2tdata.GfL.data - tdata.GbL.data)));
        label = L"Q",
        line = (:dot, :grey, 1),
    )

    twinx_ = twinx()

    # THz pulse
    p1 = plot!(
        twinx_,
        ts_plots,
        G0γ |> (-) ∘ imag ∘ diag ∘ lesser;
        label = L"n^0_\gamma",
        line = (:green, 3.0, 0.2),
        legend = :topright,
        xlims = extrema(ts_plots),
        grid = false,
        minorticks = true,
        tick_direction = :out,
        xaxis = nothing,
    )
    # xaxis=nothing)

    # Kondo timescale
	max_pulse_i = findmax(G0γ |> (-) ∘ imag ∘ diag ∘ lesser)[2]
	
    t_max_pulse = ts_plots[max_pulse_i]
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

# ╔═╡ 77a2227a-915c-11eb-00d2-2dbbd3ec938e
# Comparison of frequency *projected* solution with time solution
begin
    Gd_τ = let
        f = EchoPulse.ft(ωmesh, 1im / pi * imag(Geq[2]); mode = -1)
        plot(
            f[1],
            imag(f[2]);
            label = L"\mathcal{F}^{-1}_\omega\{A_d(\omega)\}(\tau)",
            xlabel = L"\tau \;D",
            xlims = (0.25τs[1], 0.25τs[end]),
            legend = :topright,
            alpha = 1.0,
            ms = 1.0,
        )
    end

    Gd_τ = plot!(
        τs,
        (imag ∘ diag)(reverse(-ρt; dims = 2));
        label = L"A_d(\tau)",
        ms = 2.0,
    )
	
	Gd_τ = plot!(
	τs,
	((-) ∘ imag ∘ diag)(reverse(greater(G0γ)-lesser(G0γ); dims = 2));
	label = L"A_\gamma(\tau)",
	ms = 2.0,
	)

    Gd_ω = plot(
        ωs_plots,
        imag(2Geq[2]);
        label = L"A_d(\omega) \quad\textrm{(eq)}",
        xlabel = L"\omega / D",
        xlims = (ωs[1], ωs[end]),
        legend = :topright,
        linecolor = :darkorange,
    )

    Gd_ω = plot!(
        ωs,
        real(ρω[:, max_pulse_i]);
        label = L"\mathcal{F}_\tau\{A_d(T_0,\tau)\}(\omega)",
        linecolor = :green,
    )

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

    plot(Gd_τ, Gd_ω; layout = grid(2, 1))
    # savefig("equilibrium-comparison.pdf")
end

# ╔═╡ 26ade5f5-5b4e-460e-ad4a-62a842fb63b0
begin
	plot(ts, G0γ + (G0γ ⋆ Tγ ⋆ G0γ) |> (-) ∘ imag ∘ diag ∘ lesser,
		label = L"A_\gamma(T)",
		xlabel = L"T \;D",
	)
	plot!(ts, G0γ |> (-) ∘ imag ∘ diag ∘ lesser,
		label = L"A^0_\gamma(T)"
	)
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

# ╔═╡ 7543c1d5-1d72-49de-bfae-396facea15a4
# begin
#     	fω, __ = wigner_transform_itp(0.5 * (tdata.GfG.data .* conj(-tdata.GbL.data) + tdata.GfL.data .* conj(-tdata.GbG.data)), ts; fourier=true)

#     	fω = plot(ωs, imag(fω[:,length(ts) ÷ 2]) ./ real(ρω[:,length(ts) ÷ 2]) .+ 0.5;
#     		label=L"\mathcal{F}_\tau\{f(T_0,\tau)\}(\omega)", 
#     		linecolor=:darkblue
#     	)

#     	fω = plot!(ωs_plots, 1.0 ./ (exp.(β .* ωs_plots) .+ 1); 
#     		label=L"f_{eq}(\omega)", 
#     		xlabel=L"\omega / D", xlims=(ωs[1], ωs[end]),
#     		legend=:topright,
#     		linecolor=:darkorange)
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
# ╟─292f7a80-6a1e-11eb-144e-b96c3e89f430
# ╠═49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# ╟─a8b39952-53d9-49b6-ada4-d7206640857a
# ╟─fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
# ╟─56d36398-6a32-11eb-3f78-01155850c5d2
# ╟─0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
# ╟─06020ea7-d32d-4f06-972b-a711aab1631b
# ╟─eda788ba-6f84-11eb-24ee-17e261d30bc1
# ╟─77a2227a-915c-11eb-00d2-2dbbd3ec938e
# ╟─26ade5f5-5b4e-460e-ad4a-62a842fb63b0
# ╟─0483ff48-915c-11eb-12f7-fb5549cf4d66
# ╟─afbf57c4-915f-11eb-02ec-a5e41732d549
# ╟─7543c1d5-1d72-49de-bfae-396facea15a4
# ╟─b7e0f8bc-907a-11eb-21b3-f9471bb006fb
# ╟─9a788356-912b-11eb-0356-35fb2d865d0b
