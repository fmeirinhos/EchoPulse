"""
Photon assisted single impurity Anderson Model
"""
struct PhotonAssistedSIAM{T<:AbstractFloat}
    # physical parameters
    V0::T     # Hybridization strength
    ε0f::T    # F-electron ground state e
    U::T

    ω0::T     # Photon frequency
    Ω0::T     # Photon width
    Γ0::T     # Photon hybridization strength

    # other parameters
    η::T      # particle density (poor-man's dmft)
    γ::T      # particle density (poor-man's dmft)

    a::T
    function PhotonAssistedSIAM(; V0, ε0f, U, ω0, Ω0, Γ0, γ, η, a=0)
        t = promote(V0, ε0f, U, ω0, Ω0, Γ0, η, γ, a)
        return new{eltype(t)}(t...)
    end
end

"""
Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
    GcL::Any
    GcG::Any
    dos_c::Vector{Float64}
    εs::Vector{Float64}

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

function TimeData(; model::PhotonAssistedSIAM, ωs, β, Tγ, Nγ, εs, timematrix)
    @assert isodd(length(ωs)) && first(ωs) == -last(ωs) "Must be met for double `ft` to act as identity."

    (; ω0, Ω0) = model

    ts = ft(ωs, ωs; inverse = true)[1]
    Θt = (t -> t > 0 ? 1 : t == 0 ? 0.5 : 0.0).(ts) # Heaviside magic

    GγL0_f(t, t′; Nγ=Nγ) = -im * (0 + Nγ * sech((0.5(t + t′) - Tγ) * Ω0)^2) * exp(-1im * ω0 * (t - t′)) * sech((t - t′) * Ω0/4)^2
    GγG0_f(t, t′; Nγ=Nγ) = -im * (1 + Nγ * sech((0.5(t + t′) - Tγ) * Ω0)^2) * exp(-1im * ω0 * (t - t′)) * sech((t - t′) * Ω0/4)^2

    # Calculate ργ>(t) and ργ<(t)
    GγL, GγG = let
        # photon density of states
        dos_γ(ω) = 2π * gaussian(ω, +model.ω0, model.Ω0)

        # Initially the photon field is *NOT* occupied: β → ∞
        GγL = (τ -> GγL0_f(τ/2, -τ/2; Nγ=0)).(ts)
        GγG = (τ -> GγG0_f(τ/2, -τ/2; Nγ=0)).(ts)

        GγL, GγG
    end

    # Calculate ρc>(t) and ρc<(t) from ρc(ω) = ∑k Gc(k; ω)
    ρcL, ρcG = let
        ρc(z) = 2 * gaussian(z, 0.0, 1.0) * (π * erfi(z / sqrt(2)) - log(z) - log(-1 / z)) |> imag
        ρcL = ft(ωs, +im * ρc.(ωs .- im * 1e-2) .* nf.(+ωs, β); inverse = true)[2]
        ρcG = ft(ωs, -im * ρc.(ωs .- im * 1e-2) .* nf.(-ωs, β); inverse = true)[2]

        ρcL, ρcG
    end

    # Obtain pseudo-particle Greater and Lesser functions in equilibrium
    GG, GL, GdR_eq, ρcR_eq = let
        ρc = ρcG - ρcL

        # Solve NCA equations
        Σt = fixed_point(hcat(ρc, ρc, ρc, -ρc, ρc, ρc)) do Σt
            # Σ>(t) –> ΣR(ω) & Σ<(t) –> Σ<(ω)
            ΣR = mapslices(x -> ft(ts, x .* Θt; inverse = false)[2], Σt[:, 1:3]; dims = 1)
            ΣL = mapslices(x -> ft(ts, x; inverse = false)[2], Σt[:, 4:6]; dims = 1)

            GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)

            # GR(ω) –> G>(t) & G<(ω) –> G<(t)
            GG = mapslices(x -> ft(ωs, im * 2imag(x); inverse = true)[2], GR; dims = 1)
            GL = mapslices(x -> ft(ωs, im * 1imag(x); inverse = true)[2], GL; dims = 1)

            Σt′ = calculate_Σt(model, ρcG, ρcL, GγG, GγL, GG, GL)

            return Σt′
        end

        ΣR = mapslices(x -> ft(ts, x .* Θt; inverse = false)[2], Σt[:, 1:3]; dims = 1)
        ΣL = mapslices(x -> ft(ts, x; inverse = false)[2], Σt[:, 4:6]; dims = 1)

        GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)

        GG = mapslices(x -> ft(ωs, im * 2imag(x); inverse = true)[2], GR; dims = 1)
        GL = mapslices(x -> ft(ωs, im * 1imag(x); inverse = true)[2], GL; dims = 1)

        GdR = ft(ts, (@. +im * ((GG[:, 2] * ~GL[:, 1] - GG[:, 3] * ~GL[:, 2]) - (GL[:, 2] * ~GG[:, 1] - GL[:, 3] * ~GG[:, 2])) * Θt); inverse = false)[2]

        TcR = abs2(model.V0) * GdR
        ρ0cR = ft(ts, ρc .* Θt; inverse = false)[2]

        GG, GL, GdR, ρ0cR + ρ0cR .* TcR .* ρ0cR
    end

    # Kondo temperature estimation
    ρω0 = -1 / (2π) * imag((ts[2] - ts[1]) * sum(ρcG - ρcL))
    ωK = exp(model.ε0f / (2 * abs2(model.V0) * ρω0))

    # c-electrons
    if isempty(εs)
        GcL = let
            itp = extrapolate(interpolate((ts,), ρcL, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        GcG = let
            itp = extrapolate(interpolate((ts,), ρcG, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
    elseif timematrix
        @assert false "unsupported"
    else
        # Non-interacting conduction-electron GF
        GcL = [GreenFunction(+im * nf(+ε, β) * ones(1, 1), SkewHermitian) for ε in εs]
        GcG = [GreenFunction(-im * nf(-ε, β) * ones(1, 1), SkewHermitian) for ε in εs]
    end
    dos_c = gaussian.(εs, 0.0, 1.0)

    # auxiliary-particles
    if timematrix
        # Trim the time-grid to contain 2 Kondo time-scale. Looks like 4 timescales but G∞ will "chop" it in half
        τK = inv(ωK)
        r = argmin(abs.(ts .+ 2τK)):argmin(abs.(ts .- 2τK))

        GbL = GreenFunction(G∞(GL[r, 1]), SkewHermitian)
        GbG = GreenFunction(G∞(GG[r, 1]), SkewHermitian)

        GfL = GreenFunction(G∞(GL[r, 2]), SkewHermitian)
        GfG = GreenFunction(G∞(GG[r, 2]), SkewHermitian)

        GaL = GreenFunction(G∞(GL[r, 3]), SkewHermitian)
        GaG = GreenFunction(G∞(GG[r, 3]), SkewHermitian)

        # initial time-grid (equidistant)
        t0 = ts[1:size(GfL, 1)] .- ts[size(GfL, 1)]
    else
        # find τ = 0 index (corresponds to ∫dω)
        i0 = length(ts) ÷ 2 + 1

        GbL = GreenFunction(GL[i0, 1] * ones(1, 1), SkewHermitian)
        GbG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GfL = GreenFunction(GL[i0, 2] * ones(1, 1), SkewHermitian)
        GfG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GaL = GreenFunction(GL[i0, 3] * ones(1, 1), SkewHermitian)
        GaG = GreenFunction(-im * ones(1, 1) * Int(!isinf(model.U)), SkewHermitian)

        # initial time-grid
        t0 = [0.0]
    end

    # gaussian pulse
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
    # @assert isinf(model.U) || all(isapprox.(diag(GaG.data), -im; rtol=1e-3))

    data = TimeData(GcL, GcG, dos_c, εs, GfL, GfG, GbL, GbG, GaL, GaG, GγL, GγG, GγL0, GγG0, GγL0_f, GγG0_f, zero(GfL), zero(GfG), zero(GbL), zero(GbG), zero(GaL), zero(GaG), zero(GbL), zero(GbG), zero(GγL), zero(GγG))

    for t in eachindex(t0), t′ in eachindex(t0)
        Σ!(t0, nothing, nothing, t, t′, model, data)
    end

    return (; GdR_eq, ρcR_eq, ωK, data, t0)
end
