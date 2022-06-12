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
    𝒢cL::Any    # Weiss field
    𝒢cG::Any    # Weiss field
    GcL         # Local field
    GcG         # Local field

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
    GγL0_f::Function
    GγG0_f::Function

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
end

function PhotonAssistedAIMData(; model::PhotonAssistedAIM, ωs, β, Tγ, Nγ, timematrix, dmft)
    @assert isodd(length(ωs)) && first(ωs) == -last(ωs) "Must hold for double `ft` to be an identity."

    # Discrete Fourier grid of ωs
    τs = ft(ωs, ωs; inverse = true)[1]

    # Non-interacting photon Green functions
    GγL0_f(t, t′; Nγ = Nγ) = -im * (0 + Nγ * sech((0.5(t + t′) - Tγ) * model.Ω0)^2) * exp(-1im * model.ω0 * (t - t′)) * sech((t - t′) * model.Ω0 / 4)^2
    GγG0_f(t, t′; Nγ = Nγ) = -im * (1 + Nγ * sech((0.5(t + t′) - Tγ) * model.Ω0)^2) * exp(-1im * model.ω0 * (t - t′)) * sech((t - t′) * model.Ω0 / 4)^2

    # Solve the system in thermal equilibrium
    (; GdRω, GcRω, 𝒢cRω, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ) = let
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

            # Bethe-lattice k-sum
            𝒢cRω, GcRω = let
                𝒢cRω = ft(τs, Στ[:, 7] .* (τ -> τ >= 0 ? 1 : 0.0).(τs); inverse = false)[2] # enforces Heaviside-sparsity (not guaranteed by fixed_point)
                GcRω = @. 𝒢cRω + abs2(model.V0) * 𝒢cRω * GdRω * 𝒢cRω
                𝒢cRω = dmft ? inv.(ωs - GcRω) : -im * π * bethe_dos.(ωs)
                𝒢cRω, GcRω
            end

            𝒢cRτ = ft(ωs, +2im * imag(𝒢cRω); inverse = true)[2] .* Θτ
            𝒢cGτ = ft(ωs, +2im * imag(𝒢cRω) .* nf.(-ωs, β); inverse = true)[2]
            𝒢cLτ = ft(ωs, -2im * imag(𝒢cRω) .* nf.(+ωs, β); inverse = true)[2]

            ΣGτ, ΣLτ = calculate_Στ(model, 𝒢cGτ, 𝒢cLτ, GγGτ, GγLτ, GGτ, GLτ)
            (; Στ′ = hcat(ΣGτ, ΣLτ, 𝒢cRτ), GdRω, GcRω, 𝒢cRω, 𝒢cLτ, 𝒢cGτ, GLτ, GGτ)
        end

        # Proper Heaviside
        Θτ = (τ -> τ > 0 ? 1 : τ == 0 ? 0.5 : 0.0).(τs)

        # Initially (T ≤ 0) the photon field is *NOT* occupied: β → ∞
        GγLτ = (τ -> GγL0_f(+τ / 2, -τ / 2)).(τs)
        GγGτ = (τ -> GγG0_f(+τ / 2, -τ / 2)).(τs)

        # C-electron density of states
        ρc = ft(ωs, -im * 2π * bethe_dos.(ωs); inverse = true)[2]

        # Solve equilibrium problem
        fixed_point(first ∘ eq_loop, hcat(ρc, ρc, ρc, -ρc, ρc, ρc, ρc .* Θτ)) |> eq_loop
    end

    # Assure commutation relations
    # i0 = length(τs) ÷ 2 + 1
    # GGτ[i0,1:2] .= -im

    # Kondo temperature estimation (ω = 0 corresponds to ∫dτ)
    ρω0 = -1 / (2π) * imag((τs[2] - τs[1]) * sum(𝒢cGτ - 𝒢cLτ))
    ωK = exp(model.ε0f / (2 * abs2(model.V0) * ρω0))

    if timematrix
        # Trim the time-grid to contain 2 Kondo time-scale. Looks like 4 timescales but G∞ will "chop" it in half
        τK = inv(ωK)
        r = argmin(abs.(τs .+ 2τK)):argmin(abs.(τs .- 2τK))

        GbL = GreenFunction(G∞(GLτ[r, 1]), SkewHermitian)
        GbG = GreenFunction(G∞(GGτ[r, 1]), SkewHermitian)

        GfL = GreenFunction(G∞(GLτ[r, 2]), SkewHermitian)
        GfG = GreenFunction(G∞(GGτ[r, 2]), SkewHermitian)

        GaL = GreenFunction(G∞(GLτ[r, 3]), SkewHermitian)
        GaG = GreenFunction(G∞(GGτ[r, 3]), SkewHermitian)

        # Initial time-grid (equidistant)
        t0 = τs[1:size(GfL, 1)] .- τs[size(GfL, 1)] |> collect
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
            itp = extrapolate(interpolate((τs,), 𝒢cLτ, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        𝒢cG = let
            itp = extrapolate(interpolate((τs,), 𝒢cGτ, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
    end

    # photon pulse
    GγL = GreenFunction([GγL0_f(t1, t2) for t1 in t0, t2 in t0], SkewHermitian)
    GγG = GreenFunction([GγG0_f(t1, t2) for t1 in t0, t2 in t0], SkewHermitian)

    GγL0 = GreenFunction([GγL0_f(t1, t2) for t1 in t0, t2 in t0], SkewHermitian)
    GγG0 = GreenFunction([GγG0_f(t1, t2) for t1 in t0, t2 in t0], SkewHermitian)

    # sanity checks
    @assert all(imag(diag(GbL.data)) .<= 0.0)
    @assert all(imag(diag(GfL.data)) .>= 0.0)
    @assert all(-imag(diag(GbL.data)) + 2imag(diag(GfL.data)) - imag(diag(GaL.data)) .≈ 1.0)
    # @assert all(isapprox.(diag(GfG.data), -im; rtol=1e-3))
    # @assert all(isapprox.(diag(GbG.data), -im; rtol=1e-3))
    # @assert isinf(model.U0) || all(isapprox.(diag(GaG.data), -im; rtol=1e-3))

    data = PhotonAssistedAIMData(𝒢cL, 𝒢cG, GcL, GcG, GfL, GfG, GbL, GbG, GaL, GaG, GγL, GγG, GγL0, GγG0, GγL0_f, GγG0_f, zero(GfL), zero(GfG), zero(GbL), zero(GbG), zero(GaL), zero(GaG), zero(GbL), zero(GbG), zero(GγL), zero(GγG))

    for t in eachindex(t0), t′ in eachindex(t0)
        calculate_Σt!(t0, nothing, nothing, t, t′, model, data; dmft = dmft)
    end

    return (; GdRω, GcRω, 𝒢cRω, ωK, data, t0)
end
