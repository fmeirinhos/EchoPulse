### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ ed50b1fe-6a1e-11eb-08ab-95d436eef99c
begin
    using PlutoUI
    up = @bind updatethis Button("Update")

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
        return m
    end
end;

# ╔═╡ 292f7a80-6a1e-11eb-144e-b96c3e89f430
begin
    using Revise

    using DSP, NCA, NLsolve
    using KadanoffBaym

    EchoPulse = ingredients("../src/EchoPulse.jl").EchoPulse

    using LinearAlgebra: diag
    using UnPack

    using Test

    using Plots, LaTeXStrings
end;

# ╔═╡ 49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# parameters (units of half-bandwidth)
begin
    β = 4.0e3  # initial temperature

    model = PhotonAssistedSIAM(
        # siam
        ;
        V₀ = sqrt(0.06),# hybridization strength
        ε₀f = -0.35,# d-electron energy

        # quantum light
        Γ₀ = sqrt(0.01),  # light-matter coupling
        ω₀ = 1.0e-1, # photon energy
        Ω₀ = 1.0e-1,
    )

    # pulse parameters
    Nᵧ = 5.0e0 # maximum photon occupation
    T₀ = 1.0e2 # peak arrival time

    # integration parameters
    tmax = 2.0e2

    # computation parameters
    mesh = range(-6, 6; length = 2^14)
    rtol = 1e-3 # time-stepper tolerances
end;

# ╔═╡ 89141fc4-6a2c-11eb-11d4-432a71ae7968
begin
    nca = let
        mesh_ = LinearMesh(first(mesh), last(mesh), length(mesh))

        ρc = 2π * NCA.gaussian.(mesh_, 0.0, 1.0)
        ργ = 2π * NCA.gaussian.(mesh_, +model.ω₀, model.Ω₀)

        data = PhotonAssistedSIAMData(mesh_, β, ρc, ργ)#, x -> 1 / (exp(+1 * x) - 1))

        FreqNCA(; model = model, mesh = mesh_, data = data, β = β)
    end

    # Obtain pseudo-particle Greater and Lesser functions
    imG = let
        x0 = cat(
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, -nca.data.ρc),
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, +nca.data.ρc);
            dims = 3,
        ) # initial guesses

        # Solve NCA equations
        imΣ = nlsolve(x0; method = :anderson, beta = 0.4, show_trace = :true) do imΣ′
            imG = NCA.slave_particle_imG(nca, imΣ′)
            imΣ = NCA.slave_particle_imΣ(nca, imG)
            return imΣ - imΣ′
        end.zero

        NCA.slave_particle_imG(nca, imΣ)
    end

    Gᵈ = NCA.projector(nca, imG)[2]
    ωK = NCA.kondo_temp(nca.mesh, nca.data.ρc ./ (2π), model.ε₀f, model.V₀)
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
    ωs_plots = nca.mesh.mesh # .|> ω_to_eV

    plot(
        ωs_plots,
        0.5nca.data.ρc;
        label = L"$A_c$",
        linecolor = :green,
        legend = :topleft,
        xlabel = L"$\omega/D$",
        xlims = 0.75 .* extrema(ωs_plots),
        minorticks = true,
        title = L"\textrm{Equilibrium ~spectral ~functions}",
    )

    plot!(ωs_plots, imag(Gᵈ); label = L"$A_d$", linecolor = :darkorange)

    vline!([model.ε₀f]; label = L"\varepsilon_d", line = (:dot, 1.0, :darkblue, 0.75))

    vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (:dot, 1.0, :darkred, 0.75))

    vspan!([-ωK / 2, ωK / 2]; fillcolor = :grey, alpha = 0.3, label = L"\omega_K")
end

# ╔═╡ 0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
begin
    t_data, t0 = EchoPulse.TimeData(model, nca, imG; T₀ = T₀, Nᵧ = Nᵧ, timematrix = false)

    @unpack GᶜL, GᶜG, GᶠL, GᶠG, GᵇL, GᵇG, GᵞL, GᵞG = t_data

    @time ts =
        kbsolve(
            (u, ts, t, t′) -> EchoPulse.f_vt(model, t_data, ts, t, t′, T₀),
            (u, ts, t) -> (println(" t: $(ts[t])");
            x = EchoPulse.f_vt(model, t_data, ts, t, t, T₀);
            x .- adjoint.(x,)),
            [GᶠL, GᶠG, GᵇL, GᵇG,GᵞL,GᵞG],
            (t0, tmax);
            atol = 0.0,
            rtol = rtol,
            γ = 99 / 100,
            dtini = 1e-10,
            callback = (times, t, t′) -> EchoPulse.Σ_t(model, t_data, times, t, t′),
        ).t

    # THz pulse
    GᵞL₀ = GreenFunction([t_data.GᵞL₀(t, t′) for t in ts, t′ in ts], SkewHermitian)
    GᵞG₀ = GreenFunction([t_data.GᵞG₀(t, t′) for t in ts, t′ in ts], SkewHermitian)
end;

# ╔═╡ 9a788356-912b-11eb-0356-35fb2d865d0b
# Kubo-Martin-Schwinger boundary conditions
@testset "KMS boundary conditions" begin
    # pseudo-particles checks
    @test all(diag(GᶠG.data) .≈ -1.0im)
    @test all(diag(GᵇG.data) .≈ -1.0im)

    @test maximum(+imag(GᶠL.data)) <= 1.0 # Maximum occupation is 1
    @test maximum(-imag(GᵇL.data)) <= 1.0 # Maximum occupation is 1
    @test maximum(-imag(GᶠG.data)) <= 1.0
    @test maximum(-imag(GᵇG.data)) <= 1.0

    @test minimum(+imag(GᶠL.data)) >= -1.0
    @test minimum(-imag(GᵇL.data)) >= -1.0
    @test minimum(-imag(GᶠG.data)) >= -1.0
    @test minimum(-imag(GᵇG.data)) >= -1.0

    @test all(imag(diag(-GᵇL.data + 2GᶠL.data)) .≈ 1.0)

    # photon checks
    @test all(diag(GᵞG₀.data - GᵞL₀.data) .≈ -1.0im)
    @test all(imag(diag(GᵞL₀.data)) .<= 0.0)

    # c-electron checks
    @test all([GᶜG(t, t) - GᶜL(t, t) for t in ts] .≈ -1.0im)

    # 	if iszero(model.Γ₀)
    # 		∫dω(f) = M.integrate(freqs.mesh, f, M.TrapezoidalEvenFast()) / (2π)
    # 		≅(a,b) = isapprox(a,b;atol=5e-3)

    # 		# C^<(t,t0) = C˺(t,t0)
    # 		@test all(GᶜL[t,1] ≅ +∫dω(fermi .* (-conj(Aᶜ˹[t]))) for t in 1:size(Aᶜ˹,2))
    # 		@test all(GᶠL[t,1] ≅ +∫dω(fermi .* (-conj(GᶠT[t]))) for t in 1:size(GᶠT,2))
    # 		@test all(GᵇL[t,1] ≅ -∫dω(fermi .* (-conj(GᵇT[t]))) for t in 1:size(GᶠT,2))
    # 		@test all(ΣᶠL[t,1] ≅ +∫dω(fermi .* (-conj(ΣᶠT[t]))) for t in 1:size(ΣᶠT,2))
    # 		@test all(ΣᵇL[t,1] ≅ -∫dω(fermi .* (-conj(ΣᵇT[t]))) for t in 1:size(ΣᵇT,2))

    # 		# C^>(t0,t) = C˹(t0,t)
    # 		@test all(GᶜG[1,t] ≅ -∫dω(fermi⁻ .* Aᶜ˹[t]) for t in 1:size(Aᶜ˹,2))
    # 		@test all(GᶠG[1,t] ≅ -∫dω(fermi⁻ .* GᶠT[t]) for t in 1:size(GᶠT,2))
    # 		@test all(GᵇG[1,t] ≅ -∫dω(fermi⁻ .* GᵇT[t]) for t in 1:size(GᵇT,2))
    # 		@test all(ΣᶠG[1,t] ≅ -∫dω(fermi⁻ .* ΣᶠT[t]) for t in 1:size(ΣᶠT,2))
    # 		@test all(ΣᵇG[1,t] ≅ -∫dω(fermi⁻ .* ΣᵇT[t]) for t in 1:size(ΣᵇT,2))
    # 		else
    # 		println("Cannot test mixed functions BCs")
    # 	end
end;

# ╔═╡ eda788ba-6f84-11eb-24ee-17e261d30bc1
begin
    # Spectral function
    ρt = 1im * (GᶠG.data .* conj(-GᵇL.data) - GᶠL.data .* conj(-GᵇG.data))

    ρτ, (_, Ts) = wigner_transform_itp(ρt, ts; fourier = false)
    ρω, (ωs, Ts) = wigner_transform_itp(ρt, ts; fourier = true)

    ts_plots = ts

    # Spectral function
    h1 = heatmap(
        Ts,
        ωs,
        real(ρω);
        xaxis = nothing,
        ylabel = L"\omega /D",
        ylims = (-2.0, 2.0),
        legend = :none,
        c = :magma,
        title = L"A_d(\omega,T)",
    )
    # minorticks=true,
    # tick_direction=:out)

    colorbar = scatter(
        [0, 0],
        [1, 1];
        zcolor = [0, 1],
        xaxis = nothing,
        xlims = (1, 1),
        yaxis = nothing,
        label = nothing,
        grid = nothing,
        c = :magma,
        clims = extrema(real(ρω)),
    )

    # Occupations
    p1 = plot(
        ts_plots,
        imag(diag(GᶠL.data));
        label = L"n_f",
        title = L"\textrm{Field occupations}",
        legend = :topleft,
        xlims = extrema(ts_plots),
        minorticks = true,
        tick_direction = :out,
    )
    p1 = plot!(ts_plots, imag(diag(-GᵇL.data)); label = L"n_b")
    p1 = plot!(
        ts_plots,
        imag(diag((2GᶠL.data - GᵇL.data)));
        label = L"Q",
        line = (:dot, :grey, 1),
    )

    twinx_ = twinx()

    # THz pulse
    p1 = plot!(
        twinx_,
        ts_plots,
        -imag(diag(GᵞL₀.data));
        label = L"n^0_\gamma",
        line = (:green, 3.0, 0.2),
        legend = :topright,
        xlabel = L"T\;D",
        xlims = extrema(ts_plots),
        grid = false,
        minorticks = true,
        tick_direction = :out,
    )
    # xaxis=nothing)

    # Kondo timescale
    t_max_pulse = ts_plots[findmax(-imag(diag(GᵞL₀.data)))[2]]
    p1 = vspan!(
        twinx_,
        t_max_pulse .+ 1 / ωK .* [-0.5, +0.5];
        fillcolor = :grey,
        alpha = 0.3,
        linewidth = 0,
        markersize = 0,
        label = L"T_K",
    )

    plot(h1, p1, colorbar; layout = @layout [grid(2, 1){0.9w} a])
end

# ╔═╡ 0483ff48-915c-11eb-12f7-fb5549cf4d66
begin
    # @unpack data, mesh = ω_data.nca
    Q = NCA.integrate(nca.mesh, -imG[:, 1, 2] + 2imG[:, 2, 2]) / (2π)
    ts_ft = EchoPulse.ft(mesh, zero(nca.mesh.mesh); mode = -1)[1]
    τs = ts .- reverse(ts)

    # Lesser
    L = mapslices(
        x -> EchoPulse.ft(mesh, x / (2pi); mode = -1)[2],
        1.0im * hcat(
            imG[:, :, 2] / Q,
            +nca.data.fermi .* nca.data.ρc,
            -nca.data.bose .* nca.data.ργ,
        );
        dims = 1,
    )

    # Greater
    G = mapslices(
        x -> EchoPulse.ft(mesh, x / (2pi); mode = -1)[2],
        1.0im * hcat(
            imG[:, :, 1],
            -nca.data.fermi⁻ .* nca.data.ρc,
            -nca.data.bose⁻ .* nca.data.ργ,
        );
        dims = 1,
    )

    xlims = 0.25 .* extrema(τs)
    xlabel = L"\tau D"

    GbLτ = plot(
        ts_ft,
        imag(L[:, 1]);
        label = L"\mathcal{F}_\omega\{G^<_b\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GbLτ = plot!(τs, imag(diag(reverse(GᵇL.data; dims = 2))); label = L"G^<_b(\tau)")

    GbGτ = plot(
        ts_ft,
        imag(G[:, 1]);
        label = L"\mathcal{F}_\omega\{G^>_b\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GbGτ = plot!(τs, imag(diag(reverse(GᵇG.data; dims = 2))); label = L"G^>_b(\tau)")

    GfLτ = plot(
        ts_ft,
        imag(L[:, 2]);
        label = L"\mathcal{F}_\omega\{G^<_f\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GfLτ = plot!(τs, imag(diag(reverse(GᶠL.data; dims = 2))); label = L"G^<_f(\tau)")

    GfGτ = plot(
        ts_ft,
        imag(G[:, 2]);
        label = L"\mathcal{F}_\omega\{G^>_f\}(\tau)",
        legend = :topright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GfGτ = plot!(τs, imag(diag(reverse(GᶠG.data; dims = 2))); label = L"G^>_f(\tau)")

    GcLτ = plot(
        ts_ft,
        imag(L[:, 3]);
        label = L"\mathcal{F}_\omega\{G^<_c\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GcLτ = plot!(
        τs,
        imag(diag(reverse([GᶜL(t1, t2) for t1 in ts, t2 in ts]; dims = 2)));
        label = L"G^<_c(\tau)",
    )

    GcGτ = plot(
        ts_ft,
        imag(G[:, 3]);
        label = L"\mathcal{F}_\omega\{G^>_c\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GcGτ = plot!(
        τs,
        imag(diag(reverse([GᶜG(t1, t2) for t1 in ts, t2 in ts]; dims = 2)));
        label = L"G^>_c(\tau)",
    )

    GγLτ = plot(
        ts_ft,
        imag(L[:, 4]);
        label = L"\mathcal{F}_\omega\{G^<_\gamma\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GγLτ = plot!(τs, imag(diag(reverse(GᵞL₀.data; dims = 2))); label = L"G^<_\gamma(\tau)")

    GγGτ = plot(
        ts_ft,
        imag(G[:, 4]);
        label = L"\mathcal{F}_\omega\{G^>_\gamma\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    GγGτ = plot!(τs, imag(diag(reverse(GᵞG₀.data; dims = 2))); label = L"G^>_\gamma(\tau)")

    plot(GbLτ, GbGτ, GfLτ, GfGτ, GcLτ, GcGτ, GγLτ, GγGτ; layout = @layout [a b; c d; e f; g h])
end

# ╔═╡ afbf57c4-915f-11eb-02ec-a5e41732d549
begin
    imΣ = EchoPulse.NCA.slave_particle_imΣ(nca, imG)

    cap(x) = x.data[1:length(ts), 1:length(ts)]

    ΣL = mapslices(
        x -> EchoPulse.ft(mesh, im / 2pi * x; mode = -1)[2],
        imΣ[:, :, 2] / Q;
        dims = 1,
    )
    ΣG = mapslices(x -> EchoPulse.ft(mesh, im / 2pi * x; mode = -1)[2], imΣ[:, :, 1]; dims = 1)

    ΣᵇLτ = plot(
        ts_ft,
        imag(ΣL[:, 1]);
        label = L"\mathcal{F}_\omega\{\Sigma^<_b\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    ΣᵇLτ =
        plot!(τs, imag(diag(reverse(cap(t_data.ΣᵇL); dims = 2))); label = L"\Sigma^<_b(\tau)")

    ΣᵇGτ = plot(
        ts_ft,
        imag(ΣG[:, 1]);
        label = L"\mathcal{F}_\omega\{\Sigma^>_b\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    ΣᵇGτ =
        plot!(τs, imag(diag(reverse(cap(t_data.ΣᵇG); dims = 2))); label = L"\Sigma^>_b(\tau)")

    ΣᶠLτ = plot(
        ts_ft,
        imag(ΣL[:, 2]);
        label = L"\mathcal{F}_\omega\{\Sigma^<_f\}(\tau)",
        legend = :topright,
        xlims = xlims,
        xlabel = xlabel,
    )
    ΣᶠLτ =
        plot!(τs, imag(diag(reverse(cap(t_data.ΣᶠL); dims = 2))); label = L"\Sigma^<_f(\tau)")

    ΣᶠGτ = plot(
        ts_ft,
        imag(ΣG[:, 2]);
        label = L"\mathcal{F}_\omega\{\Sigma^>_f\}(\tau)",
        legend = :bottomright,
        xlims = xlims,
        xlabel = xlabel,
    )
    ΣᶠGτ =
        plot!(τs, imag(diag(reverse(cap(t_data.ΣᶠG); dims = 2))); label = L"\Sigma^>_f(\tau)")

    plot(ΣᵇLτ, ΣᵇGτ, ΣᶠLτ, ΣᶠGτ; layout = @layout [a b; c d])
end

# ╔═╡ 77a2227a-915c-11eb-00d2-2dbbd3ec938e
# Comparison of frequency *projected* solution with time solution
begin
    Gd_τ = let
        f = EchoPulse.ft(nca.mesh.mesh, 1im / pi * imag(Gᵈ); mode = -1)
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

    Gd_τ = plot!(τs, (imag ∘ diag)(reverse(-ρt; dims = 2)); label = L"A_d(\tau)", ms = 2.0)

    Gd_ω = plot(
        ωs_plots,
        imag(2Gᵈ);
        label = L"A_d(\omega) \quad\textrm{(eq)}",
        xlabel = L"\omega / D",
        xlims = (ωs[1], ωs[end]),
        legend = :topright,
        linecolor = :darkorange,
    )

    Gd_ω = plot!(
        ωs,
        real(ρω[:, length(ts)÷2]);
        label = L"\mathcal{F}_\tau\{A_d(T_0,\tau)\}(\omega)",
        linecolor = :green,
    )

    Gd_ω = vline!([model.ε₀f]; label = L"\varepsilon_d", line = (:dot, 1.0, :darkblue, 0.75))

    Gd_ω =
        vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (:dot, 1.0, :darkred, 0.75))

    Gd_ω = vspan!([-ωK / 2, ωK / 2]; fillcolor = :grey, alpha = 0.3, label = L"\omega_K")

    plot(Gd_τ, Gd_ω; layout = grid(2, 1))
    # savefig("equilibrium-comparison.pdf")

    # 	fω, __ = wigner_transform_itp(0.5 * (GᶠG.data .* conj(-GᵇL.data) + GᶠL.data .* conj(-GᵇG.data)), ts; fourier=true)

    # 	fω = plot(ωs, imag(fω[:,length(ts) ÷ 2]) ./ real(ρω[:,length(ts) ÷ 2]) .+ 0.5;
    # 		label=L"\mathcal{F}_\tau\{f(T_0,\tau)\}(\omega)", 
    # 		linecolor=:darkblue
    # 	)

    # 	fω = plot!(ωs_plots, nca.data.fermi; 
    # 		label=L"f_{eq}(\omega)", 
    # 		xlabel=L"\omega / D", xlims=(ωs[1], ωs[end]),
    # 		legend=:topright,
    # 		linecolor=:darkorange)
end

# ╔═╡ b7e0f8bc-907a-11eb-21b3-f9471bb006fb
begin
    function plotter(t)
        # Spectral function
        ρt = GᶠL.data .* conj(-GᵇG.data) - GᶠG.data .* conj(-GᵇL.data)

        ρτ, (τs, Ts) = wigner_transform_itp(ρt, times; fourier = false)
        ρω, (ωs, Ts) = wigner_transform_itp(ρt, times; fourier = true)

        h1 = plot(
            ωs,
            imag(ρω[:, t]);
            label = L"A_d",
            title = L"\textrm{Time spectral function}",
            xlabel = L"\omega",
            ylims = extrema(imag(ρω)),
        )

        sa(t) = [
            range(0, 1; length = max(t - 5, 2))
            ones(max(1, min(5, t)))
            zeros(length(times) - t)
        ]

        # Occupations
        p1 = plot(
            ω_to_ps.(inv.(times)),
            imag ∘ diag(GᶠL.data);
            label = L"n_f",
            title = L"\textrm{Field occupations}",
            seriesalpha = sa(t),
        )
        p1 = plot!(
            ω_to_ps.(inv.(times)),
            imag ∘ diag(-GᵇL.data);
            label = L"n_b",
            seriesalpha = sa(t),
        )
        p1 = plot!(
            ω_to_ps.(inv.(times)),
            imag ∘ diag((2GᶠL.data - GᵇL.data));
            label = L"Q",
            line = (:dot, :grey, 1),
            legend = :topleft,
            xlims = ω_to_ps.(inv.(times)),
            seriesalpha = sa(t),
        )

        plt = twinx()

        p1 = plot!(
            plt,
            ω_to_ps.(inv.(times)),
            (-) ∘ imag ∘ diag(G0L);
            label = nothing,
            line = (:green, 2.5),
            legend = :topright,
            xlabel = L"T \quad (ps)",#L"T\;D",
            xlims = ω_to_ps.(inv.(times)),
            grid = false,
            seriesalpha = 0.8sa(t),
        )

        # 	# Kondo time-scale
        # 	t_max_pulse = times[findmax(G0L |> (-) ∘ imag ∘ diag)[2]]
        # 	vspan!(plt, ω_to_ps.(inv.(t_max_pulse .+ 1/ω_data.ωK .* [-0.5, +0.5]));
        # 		fillcolor=:grey,
        # 		alpha=0.3,
        # 		label=L"T_K")

        return plot(h1, p1; layout = @layout grid(2, 1))
    end

    # anim = @animate for t ∈ (length(ω_data.nca.mesh) ÷ 2):length(times)
    #     plotter(t)
    # end

    # gif(anim, "anim_fps15.gif", fps = 15)
end

# ╔═╡ 96a93a32-9161-11eb-0475-5bb24fe4c75f
# begin
# 	function integrate(x::AbstractVector, y::AbstractVector)
# 		if isone(length(x))
# 			return zero(first(y))
# 		end
# 		@inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
# 		@inbounds @fastmath @simd for i in 2:(length(y) - 1)
# 			retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
# 		end
# 		return 1//2 * retval
# 	end

# 	struct TimeOrdered end
# 	struct AntiTimeOrdered end

# 	struct OrderedGF{T}
# 		L
# 		G
# 	end

# 	function Base.getindex(G::OrderedGF{TimeOrdered}, t1::Int, t2::Int)
# 		t1 > t2 ? G.G[t1,t2] : G.L[t1,t2]
# 	end
# 	function Base.getindex(G::OrderedGF{AntiTimeOrdered}, t1::Int, t2::Int)
# 		t1 >= t2 ? G.L[t1,t2] : G.G[t1,t2]
# 	end
# 	Base.getindex(G::OrderedGF, t1::Int, ::Colon) = [G[t1,t] for t in 1:size(G.L,2)]
# 	Base.getindex(G::OrderedGF, ::Colon, t2::Int) = [G[t,t2] for t in 1:size(G.L,1)]

# 	const TO = OrderedGF{TimeOrdered}
# 	const ATO = OrderedGF{AntiTimeOrdered}

# 	function dyson_lesser1(GL, GG, TL, TG)
# 		G = copy(GL)

# 		for t in eachindex(times)			
# 			G[t,t] += integrate(times, [TO(GL,GG)[t,t′] * (
# 						+ integrate(times, TO(TL,TG)[t′,:] .* GL[:,t])
# 						- integrate(times, TL[t′,:] .* ATO(GL,GG)[:,t])
# 							) for t′ in eachindex(times)])

# 			G[t,t] -= integrate(times, [GL[t,t′] * (
# 						+ integrate(times, TG[t′,:] .* GL[:,t])
# 						- integrate(times, ATO(TL,TG)[t′,:] .* ATO(GL,GG)[:,t])
# 							) for t′ in eachindex(times)])
# 		end

# 		return diag(G)
# 	end

# 	function dyson_lesser2(g0L, g0G, tL, tG)
# 		g = copy(g0L)

# 		for t1 in eachindex(times)
# 			t2 = t1

# 			function ∫dt(i,j,A,B::Function)
# 				return sign(j-i) * integrate(times[min(i,j):max(i,j)], 
# 					[A[t1,t] * B(t) for t=min(i,j):max(i,j)])
# 			end

# 			function ∫dt(t,i,j,A,B)
# 				retval = zero(eltype(A))
# 				@fastmath @simd for k in min(i,j):(max(i,j)-1)
# 					@inbounds retval += (times[k+1] - times[k]) * (A[t,k]*B[k,t2] + A[t,k+1]*B[k+1,t2])
# 				end
# 				return 1//2 * sign(j-i) * retval
# 			end

# 			g[t1,t2] += ∫dt(1, t1, g0G, t -> begin 
# 				∫dt(t,1,t,tG,g0L) + 
# 				∫dt(t,t,t2,tL,g0L) +
# 				∫dt(t,t2,1,tL,g0G)
# 			end)
# 			g[t1,t2] += ∫dt(t2, 1, g0L, t -> begin
# 				∫dt(t,1,t2,tG,g0L) + 
# 				∫dt(t,t2,t,tG,g0G) +
# 				∫dt(t,t,1,tL,g0G)
# 			end)
# 		end
# 		return g
# 	end

# 	# NOTE: Still have to be debuged!
# 	function dyson_lesser3(G0L, G0G, TL, TG)
# 		GL = copy(G0L)

# 		for t in eachindex(times)
# 			GL[t,t] += integrate(times, [G0G[t,t′] * (
# 						+ integrate(times[1:t′], TG[t′,1:t′] .* G0L[1:t′,t])
# 						+ integrate(times[t′:t], TL[t′,t′:t] .* G0L[t′:t,t])
# 						- integrate(times, TL[t′,:] .* G0G[:,t])
# 							) for t′ in eachindex(times)])

# 			GL[t,t] -= integrate(times, [G0L[t,t′] * (
# 						+ integrate(times, TG[t′,:] .* G0L[:,t])
# 						- integrate(times[t′:t], TG[t′,t′:t] .* G0G[t′:t,t])
# 						- integrate(times[1:t′], TL[t′,1:t′] .* G0G[1:t′,t])
# 							) for t′ in eachindex(times)])
# 		end
# 		return GL
# 	end
# end;

# ╔═╡ Cell order:
# ╟─ed50b1fe-6a1e-11eb-08ab-95d436eef99c
# ╟─292f7a80-6a1e-11eb-144e-b96c3e89f430
# ╠═49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# ╟─89141fc4-6a2c-11eb-11d4-432a71ae7968
# ╟─fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
# ╟─56d36398-6a32-11eb-3f78-01155850c5d2
# ╠═0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
# ╟─9a788356-912b-11eb-0356-35fb2d865d0b
# ╟─eda788ba-6f84-11eb-24ee-17e261d30bc1
# ╟─0483ff48-915c-11eb-12f7-fb5549cf4d66
# ╟─afbf57c4-915f-11eb-02ec-a5e41732d549
# ╟─77a2227a-915c-11eb-00d2-2dbbd3ec938e
# ╟─b7e0f8bc-907a-11eb-21b3-f9471bb006fb
# ╟─96a93a32-9161-11eb-0475-5bb24fe4c75f
