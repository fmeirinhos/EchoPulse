"""
Photon-assisted Anderson impurity model
"""
Base.@kwdef mutable struct PhotonAssistedAIM{T<:AbstractFloat}
    V0::T     # Hybridization strength
    ε0f::T    # F-electron ground state energy
    U0::T     # F-electron Coulomb repulsion

    Γ0::T     # Photon hybridization strength
    ω0::T     # Photon frequency
    Ω0::T     # Photon width

    α::T = sqrt(0.1) # Coupling to the thermal bath
    α2::T = 0.1      # Coupling prefactor to the photonic bath
end

"""
Data for solving the photon-assisted Anderson impurity model in non-equilibrium
"""
struct PhotonAssistedAIMData{T<:Complex{<:AbstractFloat}}
    𝒢cL::Any#::GreenFunction{T,2,Array{T,2},SkewHermitian} # Weiss field
    𝒢cG::Any#::GreenFunction{T,2,Array{T,2},SkewHermitian} # Weiss field
    GcL::GreenFunction{T,2,Array{T,2},SkewHermitian} # Local field
    GcG::GreenFunction{T,2,Array{T,2},SkewHermitian} # Local field

    GcL_bath::Function
    GcG_bath::Function

    GfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GaL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GaG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    GγL0::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγG0::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγL0_ext::Function
    GγG0_ext::Function
    GγL0_vac::Function
    GγG0_vac::Function

    ΣfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣaL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣaG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    ΔcL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΔcG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    TγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TγG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    GdL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GdG::GreenFunction{T,2,Array{T,2},SkewHermitian}
end

function PhotonAssistedAIMData(; model::PhotonAssistedAIM, ωs, β, tγ, Nγ, timematrix, dmft, nK = 2)
    @assert isodd(length(ωs)) && first(ωs) == -last(ωs) "Must hold for double `ft` to be an identity."

    # Discrete Fourier grid of ωs
    τs = ft(ωs, ωs; inverse = true)[1]

    # Non-interacting electronic bath Green functions
    GcL_bath = let
        GγLτ = ft(ωs, +im * 2π * bethe_dos.(ωs) .* nf.(+ωs, β); inverse = true)[2]
        itp = QuadraticInterpolation(GγLτ, τs)
        (t, t′) -> itp(t - t′)
    end
    GcG_bath = let
        GγGτ = ft(ωs, -im * 2π * bethe_dos.(ωs) .* nf.(-ωs, β); inverse = true)[2]
        itp = QuadraticInterpolation(GγGτ, τs)
        (t, t′) -> itp(t - t′)
    end

    # Non-interacting vacuum photon Green functions
    cutoff = 0.1
    normalization = inv(cutoff^3 / 3)
    GγL0_vac = let
        g = model.α2 * ft(ωs, -im * 2π * (ω -> ω > 0 && ω < cutoff ? ω^2 : 0.0).(ωs) .* nb.(+ωs, β) * normalization; inverse = true)[2]
        itp = QuadraticInterpolation(g, τs)
        (t, t′) -> itp(t - t′)
    end
    GγG0_vac = let
        g = model.α2 * ft(ωs, +im * 2π * (ω -> ω > 0 && ω < cutoff ? ω^2 : 0.0).(ωs) .* nb.(-ωs, β) * normalization; inverse = true)[2]
        itp = QuadraticInterpolation(g, τs)
        (t, t′) -> itp(t - t′)
    end

    # Non-interacting external photon Green functions
    GγL0_ext(t, t′) = -im * (0 + Nγ * exp(-2 * ((t + t′) / 2 - tγ)^2 * (model.Ω0)^2)) * (2 * model.Ω0 + sqrt(2π) * (model.ω0 - im * (t - t′) / 2 * model.Ω0^2) * (2exp((model.ω0 - im * (t - t′) / 2 * model.Ω0^2)^2 / (2model.Ω0^2)) - erfcx((model.ω0 - im * (t - t′) / 2 * model.Ω0^2) / (sqrt(2) * model.Ω0)))) / (2model.Ω0 + exp(model.ω0^2 / (2 * model.Ω0^2)) * sqrt(2π) * model.ω0 * (1 + erf(model.ω0 / (sqrt(2) * model.Ω0))))
    GγG0_ext(t, t′) = -im * (1 + Nγ * exp(-2 * ((t + t′) / 2 - tγ)^2 * (model.Ω0)^2)) * (2 * model.Ω0 + sqrt(2π) * (model.ω0 - im * (t - t′) / 2 * model.Ω0^2) * (2exp((model.ω0 - im * (t - t′) / 2 * model.Ω0^2)^2 / (2model.Ω0^2)) - erfcx((model.ω0 - im * (t - t′) / 2 * model.Ω0^2) / (sqrt(2) * model.Ω0)))) / (2model.Ω0 + exp(model.ω0^2 / (2 * model.Ω0^2)) * sqrt(2π) * model.ω0 * (1 + erf(model.ω0 / (sqrt(2) * model.Ω0))))

    # Solve the system in thermal equilibrium
    (; GdRω, GcRω, 𝒢cRω, GγR0ω_ext, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ, Στ, GcGτ, GcLτ, GγL0τ_ext, GγG0τ_ext) = let

        # Proper Heaviside
        Θτ = (τ -> τ > 0 ? 1 : τ == 0 ? 0.5 : 0.0).(τs)

        # Initially (T ≤ 0) the photon field is *NOT* occupied: β → ∞
        GγL0τ_ext = (τ -> GγL0_ext(+τ / 2, -τ / 2)).(τs)
        GγG0τ_ext = (τ -> GγG0_ext(+τ / 2, -τ / 2)).(τs)
        GγL0τ_vac = (τ -> GγL0_vac(+τ / 2, -τ / 2)).(τs)
        GγG0τ_vac = (τ -> GγG0_vac(+τ / 2, -τ / 2)).(τs)

        GcRω_bath = ft(τs, ft(ωs, -im * 2π * bethe_dos.(ωs); inverse = true)[2] .* Θτ; inverse = false)[2]

        function eq_loop(Στ)
            # Σ>(τ) –> ΣR(ω) & Σ<(τ) –> Σ<(ω)
            ΣRω = mapslices(x -> ft(τs, x .* Θτ; inverse = false)[2], (@views Στ[:, 1:3]); dims = 1)
            ΣLω = mapslices(x -> ft(τs, x; inverse = false)[2], (@views Στ[:, 4:6]); dims = 1)
            ΣcRω = ft(τs, (@views Στ[:, 7]) .* (τ -> τ >= 0 ? 1 : 0.0).(τs); inverse = false)[2]

            (; GRω, GLω, 𝒢cRω, GcRω) = calculate_Gω(ωs, model, (; ΣRω, ΣLω, ΣcRω, GcRω_bath); dmft)

            # GR(ω) –> G>(τ) & G<(ω) –> G<(τ)
            GGτ = mapslices(x -> ft(ωs, +im * 2imag(x); inverse = true)[2], GRω; dims = 1)
            GLτ = mapslices(x -> ft(ωs, +im * 1imag(x); inverse = true)[2], GLω; dims = 1)

            𝒢cGτ = ft(ωs, +im * 2imag(𝒢cRω) .* nf.(-ωs, β); inverse = true)[2]
            𝒢cLτ = ft(ωs, -im * 2imag(𝒢cRω) .* nf.(+ωs, β); inverse = true)[2]

            GcGτ = ft(ωs, +im * 2imag(GcRω) .* nf.(-ωs, β); inverse = true)[2]
            GcLτ = ft(ωs, -im * 2imag(GcRω) .* nf.(+ωs, β); inverse = true)[2]

            (; ΣGτ, ΣLτ, GdGτ, GdLτ, TcGτ, TcLτ) = calculate_Στ(model, (; GGτ, GLτ, 𝒢cGτ, 𝒢cLτ, GγG0τ = GγG0τ_ext + GγG0τ_vac, GγL0τ = GγL0τ_ext + GγL0τ_vac))

            GdRω = ft(τs, (GdGτ - GdLτ) .* Θτ; inverse = false)[2]
            GγR0ω_ext = ft(τs, (GγG0τ_ext - GγL0τ_ext) .* Θτ; inverse = false)[2]

            TcRω = ft(τs, (TcGτ - TcLτ) .* Θτ; inverse = false)[2]
            ΣcRω = @. TcRω / (1 + 𝒢cRω * TcRω)
            ΣcRτ = ft(ωs, 2im * imag(ΣcRω); inverse = true)[2] .* Θτ

            (; Στ = hcat(ΣGτ, ΣLτ, ΣcRτ), GdRω, GcRω, 𝒢cRω, GγR0ω_ext, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ, GcGτ, GcLτ, GγL0τ_ext, GγG0τ_ext)
        end

        # initial guess
        x0 = ft(ωs, -im * 2π * bethe_dos.(ωs); inverse = true)[2]

        # Solve equilibrium problem
        fixed_point(first ∘ eq_loop, hcat(x0, x0, x0, -x0, x0, x0, x0 .* Θτ)) |> eq_loop
    end

    # Determine Kondo energy
    ωK, τK = let
        # ω = 0 (corresponds to ∫dτ)
        ρω0 = -1 / (2π) * imag((τs[2] - τs[1]) * sum(𝒢cGτ - 𝒢cLτ))

        D = 2 # half-bandwidth of a Bethe lattice
        N = 2 # spin degeneracy
        W0 = model.V0^2 * ρω0
        ωK = D * (N * W0 / D)^(1 / N) * exp(model.ε0f / (N * W0))
        (; ωK, τK = inv(ωK))
    end

    # Determine size of the time-matrix to consider
    if timematrix
        # Trim the time-grid to contain 4 Kondo time-scales
        r = let
            # This is clamped because if ωK is too small we can run out of memory
            τK_approx = inv(clamp(ωK, 2e-3, ωK))
            if τK_approx < τK
                @warn "Under-estimated initial matrix by $(1 - τK_approx/τK)"
            end
            argmin(abs.(τs .+ nK * τK_approx)):1:argmin(abs.(τs .- nK * τK_approx))
        end
    else
        # Find τ = 0 index (corresponds to ∫dω)
        r = argmin(abs.(τs)):argmin(abs.(τs))
    end

    GcL = GreenFunction(G∞(GcLτ[r]), SkewHermitian)
    GcG = GreenFunction(G∞(GcGτ[r]), SkewHermitian)
    if dmft
        if timematrix
            𝒢cL = GreenFunction(G∞(𝒢cLτ[r]), SkewHermitian)
            𝒢cG = GreenFunction(G∞(𝒢cGτ[r]), SkewHermitian)
        else
            @assert false "DMFT without timematrix NOT supported!"
        end
    else
        𝒢cL = let
            itp = QuadraticInterpolation(𝒢cLτ, τs)
            (t, t′) -> itp(t - t′)
        end
        𝒢cG = let
            itp = QuadraticInterpolation(𝒢cGτ, τs)
            (t, t′) -> itp(t - t′)
        end
    end

    GbL = GreenFunction(G∞(GLτ[r, 1]), SkewHermitian)
    GbG = GreenFunction(G∞(GGτ[r, 1]), SkewHermitian)

    GfL = GreenFunction(G∞(GLτ[r, 2]), SkewHermitian)
    GfG = GreenFunction(G∞(GGτ[r, 2]), SkewHermitian)

    GaL = GreenFunction(G∞(GLτ[r, 3]), SkewHermitian)
    GaG = GreenFunction(G∞(GGτ[r, 3]), SkewHermitian)

    GγL0 = GreenFunction(G∞(GγL0τ_ext[r]), SkewHermitian)
    GγG0 = GreenFunction(G∞(GγG0τ_ext[r]), SkewHermitian)

    # Initial (equidistant) time-grid, ending at t = 0.0
    t0 = range(stop = 0.0, length = size(GaL, 1), step = τs[2] - τs[1]) |> collect
    @assert length(t0) == size(GbL, 1)

    data = PhotonAssistedAIMData(𝒢cL, 𝒢cG, GcL, GcG, GcL_bath, GcG_bath, GfL, GfG, GbL, GbG, GaL, GaG, GγL0, GγG0, GγL0_ext, GγG0_ext, GγL0_vac, GγG0_vac, zero(GfL), zero(GfG), zero(GbL), zero(GbG), zero(GaL), zero(GaG), zero(GaL), zero(GaG), zero(GbL), zero(GbG), zero(GbL), zero(GbG), zero(GfL), zero(GfG))

    # Initialize self-energies
    for t1 in eachindex(t0), t2 in eachindex(t0)
        calculate_Σt!(t0, nothing, nothing, t1, t2, model, data; dmft = dmft)
    end

    @testset "Sanity checks" begin
        @testset "Occupations" begin
            @test imag(last(GbL.data)) <= 0.0
            @test imag(last(GfL.data)) >= 0.0
            @test imag(last(-GbL.data + 2GfL.data - GaL.data)) ≈ 1.0
        end

        @testset "Normalisation" begin
            @test last(GfG.data) ≈ -im atol = 5e-3
            @test last(GbG.data) ≈ -im atol = 5e-3
            isinf(model.U0) || @test last(GaG.data) ≈ -im atol = 5e-3
        end

        @testset "Self-energies" begin
            @test data.ΣbG.data ≈ G∞(Στ[r, 1]) atol = 5e-7
            @test data.ΣfG.data ≈ G∞(Στ[r, 2]) atol = 5e-7
            @test data.ΣaG.data ≈ G∞(Στ[r, 3]) atol = 5e-7
            @test data.ΣbL.data ≈ G∞(Στ[r, 4]) atol = 5e-7
            @test data.ΣfL.data ≈ G∞(Στ[r, 5]) atol = 5e-7
            @test data.ΣaL.data ≈ G∞(Στ[r, 6]) atol = 5e-7
        end

        if timematrix
            @testset "Occupations rhs" begin
                # These are the hardest to fulfill since they require the initial time-matrix
                # to be good enough to substitute the use of mixed Green functions.
                # If not satisfied, it usually suffices to increase the *boundaries* of the frequency grid
                # since they control how fine Δτ is. If the temperature is very low, it is also a good idea
                # to increase the number of Kondo time-scales to include in the initial time-matrix
                @test sum(~Στ[r, 1] .* GLτ[r, 1] - ~Στ[r, 4] .* GGτ[r, 1]) * (τs[2] - τs[1]) ≈ 0.0 atol = 5e-7
                @test sum(~Στ[r, 2] .* GLτ[r, 2] - ~Στ[r, 5] .* GGτ[r, 2]) * (τs[2] - τs[1]) ≈ 0.0 atol = 5e-7
            end
        end
    end

    return (; GdRω, GcRω, 𝒢cRω, GγR0ω_ext, ωK, τK, data, t0)
end

function resize_Σ!(data::PhotonAssistedAIMData; dmft, resize = false)
    if (n = size(data.GfL, 2)) > size(data.ΣfL, 2) || resize
        resize!(data.ΣbL, n)
        resize!(data.ΣbG, n)
        resize!(data.ΣfL, n)
        resize!(data.ΣfG, n)
        resize!(data.ΣaL, n)
        resize!(data.ΣaG, n)

        resize!(data.ΔcL, n)
        resize!(data.ΔcG, n)
        resize!(data.TcL, n)
        resize!(data.TcG, n)

        resize!(data.GγL0, n)
        resize!(data.GγG0, n)
        resize!(data.TγL, n)
        resize!(data.TγG, n)

        resize!(data.GdL, n)
        resize!(data.GdG, n)
    end
end

function initial_condition(data::PhotonAssistedAIMData; dmft)
    u0 = [data.GbL, data.GbG, data.GfL, data.GfG, data.GaL, data.GaG]
    return dmft ? [u0; [data.𝒢cL, data.𝒢cG, data.GcL, data.GcG]] : u0
end
