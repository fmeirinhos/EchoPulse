"""
Photon-assisted Anderson (impurity) model
"""
Base.@kwdef mutable struct PhotonAssistedModel{T<:AbstractFloat}
    V0::T           # Hybridization strength
    ε0f::T          # f-electron ground state energy
    U0::T     = Inf # f-electron Coulomb repulsion

    α1::T     = 0.0 # Coupling to the thermal bath
    α2::T     = 0.0 # Coupling to the photonic bath
    cutoff::T = 0.0 # Photonic bath cutoff

    Γ0::T           # Photon hybridization strength
    ω0::T           # Photon central frequency
    Ω0::T           # Photon spectral width

    Nγ::T           # Photon pulse maximum average occupation
    tγ::T           # Photon pulse peak time

    N::T      = 2.0 # Spin degeneracy

    dmft::Bool      # Whether to solve single-impurity or DMFT
end

"""
Non-interacting external photon Green functions

To good approximation, these are a circle in the (t, t') plane.
It is only approximate because in order to avoid negative and finite-density zero
photon frequencies, the function in the difference of time `(t - t')` direction 
is obtained by the modified half-Fourier transform ∫₀∞ dω ω e^{-i ω (t - t')} of
the Gaussian in centre-of-mass-time (t + t′) / 2.

There is still some clarification to go regarding whether a physical pulse can
be further distorted in the (t - t′) direction.
"""
GγL0_ext(t, t′; model) = (x = (model.ω0 - im * model.Ω0^2 * (t - t′) / 2) / model.Ω0; -im * (0 + model.Nγ * exp(-((t + t′) / 2 - model.tγ)^2 * model.Ω0^2)) * (1 + sqrt(pi) * x * (2exp(x^2) - erfcx(x))) / (1 + exp((model.ω0 / model.Ω0)^2) * sqrt(π) * model.ω0 / model.Ω0 * (1 + erf(model.ω0 / model.Ω0))))
GγG0_ext(t, t′; model) = (x = (model.ω0 - im * model.Ω0^2 * (t - t′) / 2) / model.Ω0; -im * (1 + model.Nγ * exp(-((t + t′) / 2 - model.tγ)^2 * model.Ω0^2)) * (1 + sqrt(pi) * x * (2exp(x^2) - erfcx(x))) / (1 + exp((model.ω0 / model.Ω0)^2) * sqrt(π) * model.ω0 / model.Ω0 * (1 + erf(model.ω0 / model.Ω0))))

"""
Data for solving the photon-assisted Anderson impurity model in non-equilibrium
"""
struct PhotonAssistedModelData{T<:Complex{<:AbstractFloat}}
    𝒢cL::GreenFunction{T,2,Array{T,2},SkewHermitian} # Weiss field
    𝒢cG::GreenFunction{T,2,Array{T,2},SkewHermitian} # Weiss field
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

    GγL0_ext::Function
    GγG0_ext::Function
    GγL_bath::Function
    GγG_bath::Function

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

"""
Creates the PhotonAssistedModelData

# Parameters
    - `data_eq`: The named-tuple return of `solve_eq`
    - `model::PhotonAssistedModel`: A struct with model parameters
    - `timematrix::Bool`: Whether to use timematrix initial conditions
    - `nτK::Int`: The number of Kondo timescales to include in the timematrix
    - `nmax::Int`: The maximum number of points in the initial timematrix
    - `rtol::Number`: The relative tolerance for the sanity tests
"""
function PhotonAssistedModelData(; data_eq, model::PhotonAssistedModel, timematrix, nτK, nmax = 6000, rtol, atol = 1e-3 * rtol)
    (; 𝒢cLτ, 𝒢cGτ, GcGτ, GcLτ, GLτ, GGτ, GcLτ_bath, GcGτ_bath, GγLτ_bath, GγGτ_bath, Στ, τs, τK) = data_eq

    # Trim the initial timematrix to contain `nτK` Kondo timescales `τK` or find τ = 0 index (corresponds to ∫dω)
    r = let
        # This is clamped to avoid running out of memory (5000-sized complex matrix of Float64s corresponds to ~500MB)
        T = timematrix ? clamp(nτK * τK, nτK * τK, nmax * (τs[2] - τs[1])) : 0

        if T < nτK * τK
            @warn "Under-estimated initial matrix by $(1 - T / (nτK * τK))"
        end
        argmin(abs.(τs .+ T / 2)):argmin(abs.(τs .- T / 2))
    end

    GcL = GreenFunction(G∞(GcLτ[r]), SkewHermitian)
    GcG = GreenFunction(G∞(GcGτ[r]), SkewHermitian)
    if model.dmft
        if timematrix
            𝒢cL = GreenFunction(G∞(𝒢cLτ[r]), SkewHermitian)
            𝒢cG = GreenFunction(G∞(𝒢cGτ[r]), SkewHermitian)
        else
            @assert false "DMFT without timematrix NOT supported!"
        end
    else
        𝒢cL = let
            itp = extrapolate(interpolate((τs,), 𝒢cLτ, Gridded(Linear())), 0.0)
            (t, t′) -> itp(t - t′)
        end
        𝒢cG = let
            itp = extrapolate(interpolate((τs,), 𝒢cGτ, Gridded(Linear())), 0.0)
            (t, t′) -> itp(t - t′)
        end
    end

    GbL = GreenFunction(G∞(GLτ[r, 1]), SkewHermitian)
    GbG = GreenFunction(G∞(GGτ[r, 1]), SkewHermitian)

    GfL = GreenFunction(G∞(GLτ[r, 2]), SkewHermitian)
    GfG = GreenFunction(G∞(GGτ[r, 2]), SkewHermitian)

    GaL = GreenFunction(G∞(GLτ[r, 3]), SkewHermitian)
    GaG = GreenFunction(G∞(GGτ[r, 3]), SkewHermitian)

    GcL_bath = let
        itp = extrapolate(interpolate((τs,), GcLτ_bath, Gridded(Linear())), 0.0)
        (t, t′) -> itp(t - t′)
    end
    GcG_bath = let
        itp = extrapolate(interpolate((τs,), GcGτ_bath, Gridded(Linear())), 0.0)
        (t, t′) -> itp(t - t′)
    end

    GγL_bath = let
        itp = extrapolate(interpolate((τs,), GγLτ_bath, Gridded(Linear())), 0.0)
        (t, t′) -> itp(t - t′)
    end
    GγG_bath = let
        itp = extrapolate(interpolate((τs,), GγGτ_bath, Gridded(Linear())), 0.0)
        (t, t′) -> itp(t - t′)
    end

    # Initial (equidistant) time-grid, ending at t = 0.0
    t0 = range(stop = 0.0, length = size(GaL, 1), step = τs[2] - τs[1]) |> collect
    @assert length(t0) == size(GbL, 1)

    data = PhotonAssistedModelData(𝒢cL, 𝒢cG, GcL, GcG, GcL_bath, GcG_bath, GfL, GfG, GbL, GbG, GaL, GaG, (t, t′) -> GγL0_ext(t, t′; model = model), (t, t′) -> GγG0_ext(t, t′; model = model), GγL_bath, GγG_bath, zero(GfL), zero(GfG), zero(GbL), zero(GbG), zero(GaL), zero(GaG), zero(GaL), zero(GaG), zero(GbL), zero(GbG), zero(GbL), zero(GbG), zero(GfL), zero(GfG))

    # Initialize self-energies
    for t1 in eachindex(t0), t2 in eachindex(t0)
        calculate_Σt!(t0, nothing, nothing, t1, t2, model, data)
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
            @test data.ΣbG.data ≈ G∞(Στ[r, 1]) rtol = rtol atol = atol
            @test data.ΣfG.data ≈ G∞(Στ[r, 2]) rtol = rtol atol = atol
            @test data.ΣaG.data ≈ G∞(Στ[r, 3]) rtol = rtol atol = atol
            @test data.ΣbL.data ≈ G∞(Στ[r, 4]) rtol = rtol atol = atol
            @test data.ΣfL.data ≈ G∞(Στ[r, 5]) rtol = rtol atol = atol
            @test data.ΣaL.data ≈ G∞(Στ[r, 6]) rtol = rtol atol = atol
        end

        if timematrix
            @testset "Occupations rhs" begin
                # These are the hardest to fulfill since they require the initial timematrix
                # to be good enough to substitute the use of mixed Green functions.
                # If not satisfied, it usually suffices to increase the *boundaries* of the frequency grid
                # since they control how fine Δτ is. If the temperature is very low, it is also a good idea
                # to increase the number `nK` of Kondo time-scales included in the initial timematrix
                @test sum(~Στ[r, 1] .* GLτ[r, 1] - ~Στ[r, 4] .* GGτ[r, 1]) * (τs[2] - τs[1]) ≈ sum(~Στ[:, 1] .* GLτ[:, 1] - ~Στ[:, 4] .* GGτ[:, 1]) * (τs[2] - τs[1]) rtol = rtol atol = atol
                @test sum(~Στ[r, 2] .* GLτ[r, 2] - ~Στ[r, 5] .* GGτ[r, 2]) * (τs[2] - τs[1]) ≈ sum(~Στ[:, 2] .* GLτ[:, 2] - ~Στ[:, 5] .* GGτ[:, 2]) * (τs[2] - τs[1]) rtol = rtol atol = atol
            end
        end
    end

    return (; data, t0)
end

function resize_Σ!(data::PhotonAssistedModelData; resize = false)
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

        resize!(data.TγL, n)
        resize!(data.TγG, n)

        resize!(data.GdL, n)
        resize!(data.GdG, n)
    end
end
