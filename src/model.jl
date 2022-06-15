"""
Photon-assisted Anderson impurity model
"""
Base.@kwdef struct PhotonAssistedAIM{T<:AbstractFloat}
    V0::T     # Hybridization strength
    ε0f::T    # F-electron ground state energy
    U0::T     # F-electron Coulomb repulsion

    Γ0::T     # Photon hybridization strength
    ω0::T     # Photon frequency
    Ω0::T     # Photon width
end

"""
Data for solving the photon-assisted Anderson impurity model in non-equilibrium
"""
struct PhotonAssistedAIMData{T<:Complex{<:AbstractFloat}}
    𝒢cL # Weiss field
    𝒢cG # Weiss field
    GcL # Local field
    GcG # Local field

    GfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GaL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GaG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    GγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγG::GreenFunction{T,2,Array{T,2},SkewHermitian}
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

    TcL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TγG::GreenFunction{T,2,Array{T,2},SkewHermitian}

    GdL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GdG::GreenFunction{T,2,Array{T,2},SkewHermitian}
end

function PhotonAssistedAIMData(; model::PhotonAssistedAIM, ωs, β, tγ, Nγ, timematrix, dmft, nK=2)
    @assert isodd(length(ωs)) && first(ωs) == -last(ωs) "Must hold for double `ft` to be an identity."

    # Discrete Fourier grid of ωs
    τs = ft(ωs, ωs; inverse = true)[1]

    # Non-interacting vacuum photon Green functions
    ργ_vac(ω; ω_cutoff=1.0) = 0.0 #ω > 0 && ω < ω_cutoff ? ω^2 / (ω_cutoff^3 / 3) : 0.0 # normalised
    GγL0_vac = let
        GγLτ = ft(ωs, -im * ργ_vac.(ωs) .* nb.(+ωs, β); inverse = true)[2]
        itp = QuadraticInterpolation(GγLτ, τs)
        (t, t′) -> itp(t - t′)
    end
    GγG0_vac = let
        GγGτ = ft(ωs, +im * ργ_vac.(ωs) .* nb.(-ωs, β); inverse = true)[2]
        itp = QuadraticInterpolation(GγGτ, τs)
        (t, t′) -> itp(t - t′)
    end

    # Non-interacting external photon Green functions
    GγL0_ext(t, t′) = -im * (0 + Nγ * sech((0.5(t + t′) - tγ) * model.Ω0)^2) * exp(-1im * model.ω0 * (t - t′)) * sech((t - t′) * model.Ω0 / 4)^2
    GγG0_ext(t, t′) = -im * (1 + Nγ * sech((0.5(t + t′) - tγ) * model.Ω0)^2) * exp(-1im * model.ω0 * (t - t′)) * sech((t - t′) * model.Ω0 / 4)^2

    # Solve the system in thermal equilibrium
    (; GdRω, GcRω, 𝒢cRω, GγRω, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ, Στ) = let
        function eq_loop(Στ)
            @views ΣGτ, ΣLτ = Στ[:, 1:3], Στ[:, 4:6]

            # Σ>(τ) –> ΣR(ω) & Σ<(τ) –> Σ<(ω)
            ΣRω = mapslices(x -> ft(τs, x .* Θτ; inverse = false)[2], ΣGτ; dims = 1)
            ΣLω = mapslices(x -> ft(τs, x; inverse = false)[2], ΣLτ; dims = 1)

            GRω, GLω = calculate_Gω(ωs, model, ΣRω, ΣLω)

            # GR(ω) –> G>(τ) & G<(ω) –> G<(τ)
            GGτ = mapslices(x -> ft(ωs, im * 2imag(x); inverse = true)[2], GRω; dims = 1)
            GLτ = mapslices(x -> ft(ωs, im * 1imag(x); inverse = true)[2], GLω; dims = 1)

            # Projection
            GdRω = ft(τs, (@. +im * ((GGτ[:, 2] * ~GLτ[:, 1] - GGτ[:, 3] * ~GLτ[:, 2]) - (GLτ[:, 2] * ~GGτ[:, 1] - GLτ[:, 3] * ~GGτ[:, 2])) * Θτ); inverse = false)[2]
            GdγRω = ft(τs, (@. +im * ((GGτ[:, 2] * ~GLτ[:, 1] - GGτ[:, 3] * ~GLτ[:, 2]) * (im * (GγGτ + ~GγLτ)) - (GLτ[:, 2] * ~GGτ[:, 1] - GLτ[:, 3] * ~GGτ[:, 2]) * (im * (~GγGτ + GγLτ))) * Θτ); inverse = false)[2]

            # Bethe-lattice k-sum
            (; 𝒢cRω, GcRω, ΣcRτ) = let
                ΣcR = ft(τs, Στ[:, 7] .* (τ -> τ >= 0 ? 1 : 0.0).(τs); inverse = false)[2]
                ΣcR[imag(ΣcR) .> 0.0] .= real(ΣcR[imag(ΣcR) .> 0.0]) .- im * 1e-12

                GcRω = let
                  x = ωs - ΣcR
                  @. 1/2 * (1 - sqrt(1 - 4/x^2)) * x
                end
                𝒢cRω = dmft ? inv.(ωs - GcRω) : -im * π * bethe_dos.(ωs)

                TcR = @. (model.V0^2 * GdRω + model.Γ0^2 * GdγRω)
                ΣcR = @. TcR / (1 + 𝒢cRω * TcR)
                ΣcRτ = ft(ωs, 2im * imag(ΣcR); inverse = true)[2] .* Θτ

                (; 𝒢cRω, GcRω, ΣcRτ)
            end
            # GcRω, 𝒢cRω, ΣcRτ = let
            #     𝒢cRω = ft(τs, Στ[:, 7] .* (τ -> τ >= 0 ? 1 : 0.0).(τs); inverse = false)[2] # enforces Heaviside-sparsity (not guaranteed by fixed_point)
            #     𝒢cRω[imag(𝒢cRω) .> 0.0] .= real(𝒢cRω[imag(𝒢cRω) .> 0.0]) .- im * 1e-12

            #     GcRω = @. 𝒢cRω + 𝒢cRω * (model.V0^2 * GdRω) * 𝒢cRω
            #     𝒢cRω = dmft ? inv.(ωs - GcRω) : -im * π * bethe_dos.(ωs)
            #     𝒢cRτ = ft(ωs, +2im * imag(𝒢cRω); inverse = true)[2] .* Θτ
            #     GcRω, 𝒢cRω, 𝒢cRτ
            # end

            𝒢cGτ = ft(ωs, +2im * imag(𝒢cRω) .* nf.(-ωs, β); inverse = true)[2]
            𝒢cLτ = ft(ωs, -2im * imag(𝒢cRω) .* nf.(+ωs, β); inverse = true)[2]

            ΣGτ, ΣLτ = calculate_Στ(model, 𝒢cGτ, 𝒢cLτ, GγGτ, GγLτ, GGτ, GLτ)

            (; Στ = hcat(ΣGτ, ΣLτ, ΣcRτ), GdRω, GcRω, 𝒢cRω, GγRω, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ)
        end

        # Proper Heaviside
        Θτ = (τ -> τ > 0 ? 1 : τ == 0 ? 0.5 : 0.0).(τs)

        # Initially (T ≤ 0) the photon field is *NOT* occupied: β → ∞
        GγLτ = (τ -> GγL0_vac(+τ / 2, -τ / 2) + GγL0_ext(+τ / 2, -τ / 2)).(τs)
        GγGτ = (τ -> GγG0_vac(+τ / 2, -τ / 2) + GγG0_ext(+τ / 2, -τ / 2)).(τs)
        GγRω = ft(τs, (GγGτ - GγLτ) .* Θτ; inverse = false)[2]

        # C-electron density of states
        ρc = ft(ωs, -im * 2π * bethe_dos.(ωs); inverse = true)[2]

        # Solve equilibrium problem
        fixed_point(first ∘ eq_loop, hcat(ρc, ρc, ρc, -ρc, ρc, ρc, ρc .* Θτ)) |> eq_loop
    end
    
    ωK = let
        # ω = 0 (corresponds to ∫dτ)
        ρω0 = -1 / (2π) * imag((τs[2] - τs[1]) * sum(𝒢cGτ - 𝒢cLτ))

        D = 2 # half-bandwidth of a Bethe lattice
        N = 2 # spin degeneracy
        W0 = model.V0^2 * ρω0
        D * (N * W0 / D)^(1 / N) * exp(model.ε0f / (N * W0))
    end
    τK = inv(ωK)

    if timematrix
        # Trim the time-grid to contain 4 Kondo time-scales
        r = let
            # This is clamped because if ωK is too small we can run out of memory
            τK_approx = inv(clamp(ωK, 3e-3, ωK))
            if τK_approx < τK
                @warn "Under-estimated initial matrix by $(1 - τK_approx/τK)"
            end
            argmin(abs.(τs .+ nK * τK_approx)):1:argmin(abs.(τs .- nK * τK_approx))
        end

        GbL = GreenFunction(G∞(GLτ[r, 1]), SkewHermitian)
        GbG = GreenFunction(G∞(GGτ[r, 1]), SkewHermitian)

        GfL = GreenFunction(G∞(GLτ[r, 2]), SkewHermitian)
        GfG = GreenFunction(G∞(GGτ[r, 2]), SkewHermitian)

        GaL = GreenFunction(G∞(GLτ[r, 3]), SkewHermitian)
        GaG = GreenFunction(G∞(GGτ[r, 3]), SkewHermitian)

        # Initial (equidistant) time-grid, ending at t = 0.0
        t0 = range(stop = 0.0, length=size(GaL,1), step=τs[2] - τs[1]) |> collect

        @assert length(t0) == size(GfL, 1)
    else
        # Find τ = 0 index (corresponds to ∫dω)
        i0 = length(τs) ÷ 2 + 1

        GbL = GreenFunction(GLτ[i0, 1] * ones(1, 1), SkewHermitian)
        GbG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GfL = GreenFunction(GLτ[i0, 2] * ones(1, 1), SkewHermitian)
        GfG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GaL = GreenFunction(GLτ[i0, 3] * ones(1, 1), SkewHermitian)
        GaG = GreenFunction(-im * ones(1, 1) * Int(!isinf(model.U0)), SkewHermitian)

        # Initial time-grid
        t0 = [0.0]
    end

    if dmft
        if timematrix
            𝒢cL = GreenFunction(G∞(𝒢cLτ[r]), SkewHermitian)
            𝒢cG = GreenFunction(G∞(𝒢cGτ[r]), SkewHermitian)

            GcL = let
                GcLτ = ft(ωs, -2im * imag(GcRω) .* nf.(+ωs, β); inverse = true)[2]
                GreenFunction(G∞(GcLτ[r]), SkewHermitian)
            end
            GcG = let
                GcGτ = ft(ωs, +2im * imag(GcRω) .* nf.(-ωs, β); inverse = true)[2]
                GreenFunction(G∞(GcGτ[r]), SkewHermitian)
            end
        else
            @assert false "DMFT without timematrix NOT supported!"
        end
    else
        𝒢cL = let
            # itp = extrapolate(interpolate((τs,), 𝒢cLτ, Gridded(Linear())), 0)
            itp = QuadraticInterpolation(𝒢cLτ, τs)
            (t, t′) -> itp(t - t′)
        end
        𝒢cG = let
            # itp = extrapolate(interpolate((τs,), 𝒢cGτ, Gridded(Linear())), 0)
            itp = QuadraticInterpolation(𝒢cGτ, τs)
            (t, t′) -> itp(t - t′)
        end

        GcL = GreenFunction(zeros(ComplexF64, 1, 1), SkewHermitian)
        GcG = GreenFunction(zeros(ComplexF64, 1, 1), SkewHermitian)
    end

    # photon pulse
    GγL = GreenFunction(GγL0_vac.(t0', t0), SkewHermitian)
    GγG = GreenFunction(GγG0_vac.(t0', t0), SkewHermitian)

    GγL0 = GreenFunction(GγL0_vac.(t0', t0), SkewHermitian)
    GγG0 = GreenFunction(GγG0_vac.(t0', t0), SkewHermitian)

    data = PhotonAssistedAIMData(𝒢cL, 𝒢cG, GcL, GcG, GfL, GfG, GbL, GbG, GaL, GaG, GγL, GγG, GγL0, GγG0, GγL0_ext, GγG0_ext, GγL0_vac, GγG0_vac, zero(GfL), zero(GfG), zero(GbL), zero(GbG), zero(GaL), zero(GaG), zero(GbL), zero(GbG), zero(GγL), zero(GγG), zero(GfL), zero(GfG))

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
            @test last(GfG.data) ≈ -im atol=5e-3
            @test last(GbG.data) ≈ -im atol=5e-3
            isinf(model.U0) || @test last(GaG.data) ≈ -im atol=5e-3
        end

        if timematrix
            @testset "Self-energies" begin
                @test data.ΣbG.data ≈ G∞(Στ[r,1])
                @test data.ΣfG.data ≈ G∞(Στ[r,2])
                @test data.ΣaG.data ≈ G∞(Στ[r,3])
                @test data.ΣbL.data ≈ G∞(Στ[r,4])
                @test data.ΣfL.data ≈ G∞(Στ[r,5])
                @test data.ΣaL.data ≈ G∞(Στ[r,6])
            end

            @testset "Occupations rhs" begin
                # These are the hardest to fulfill since they require the initial time-matrix
                # to be good enough to substitute the use of mixed Green functions.
                # If not satisfied, it usually suffices to increase the *boundaries* of the frequency grid
                # since they control how fine Δτ is. If the temperature is very low, it is also a good idea
                # to increase the number of Kondo time-scales to include in the initial time-matrix
                @test sum(~Στ[r,1] .* GLτ[r,1] - ~Στ[r,4] .* GGτ[r,1]) * (τs[2] - τs[1]) ≈ 0.0 atol=5e-7
                @test sum(~Στ[r,2] .* GLτ[r,2] - ~Στ[r,5] .* GGτ[r,2]) * (τs[2] - τs[1]) ≈ 0.0 atol=5e-7
            end
        end
    end

    return (; GdRω, GcRω, 𝒢cRω, GγRω, ωK, τK, data, t0)
end
