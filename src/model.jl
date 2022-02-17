"""
Photon assisted single impurity Anderson Model
"""
struct PhotonAssistedSIAM{T<:AbstractFloat}
  # physical parameters
  V₀::T     # Hybridization strength
  ε₀f::T    # F-electron ground state e

  ω₀::T     # Photon frequency
  Ω₀::T     # Photon width
  Γ₀::T     # Photon hybridization strength

  # other parameters
  η::T      # c-electron artificial damping (solves a weird behaviour at large `τ` after summing all frequencies)
  γ::T      # particle density (poor-man's dmft)

  function PhotonAssistedSIAM(; V₀, ε₀f, ω₀, Ω₀, Γ₀, γ, η=0.03)
    t = promote(V₀, ε₀f, ω₀, Ω₀, Γ₀, η, γ)
    return new{eltype(t)}(t...)
  end
end

"""
Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
    GcL::Any
    GcG::Any
    dos_c::Union{Nothing,Vector{Float64}}
    εs::Vector{Float64}

    GfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγL0::Function
    GγG0::Function

    ΣfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TγG::GreenFunction{T,2,Array{T,2},SkewHermitian}
end

function TimeData(; model::PhotonAssistedSIAM, ωs, β, T₀, Nᵧ, εs, dmft, timematrix)
    # Calculate ρc>(t) and ρc<(t) from ρc(ω) = ∑k Gc(k; ω)
    ρcL, ρcG = let
        ρc(z) = 2 * gaussian(z, 0.0, 1.0) * (π * erfi(z / sqrt(2)) - log(z) - log(-1/z)) |> imag
        ρcL = ft(ωs, +im / (2π) * ρc.(ωs .- im * model.η) .* nf.(+ωs, β); mode = -1)[2]
        ρcG = ft(ωs, -im / (2π) * ρc.(ωs .- im * model.η) .* nf.(-ωs, β); mode = -1)[2]

        ρcL, ρcG
    end
    
    # Calculate ργ>(t) and ργ<(t)
    GγL, GγG = let
        # photon density of states
        dos_γ(ω) = 2π * gaussian(ω, +model.ω₀, model.Ω₀)

        # Initially the photon field is *NOT* occupied
        GγL = ft(ωs, -0im / (2π) * dos_γ.(ωs); mode = -1)[2]
        GγG = ft(ωs, -1im / (2π) * dos_γ.(ωs); mode = -1)[2]

        GγL, GγG
    end

    ts = ft(ωs, ωs; mode= -1)[1]
    Θt = (t -> t > 0 ? 1 : t == 0 ? 0.5 : 0.0).(ts) # Heaviside magic

    # Obtain pseudo-particle Greater and Lesser functions in equilibrium
    GG, GL, Gd, Gc = let
        ρc = ρcG - ρcL

        # Solve NCA equations
        Σt = fixed_point(hcat(ρc, ρc, ρc, -ρc)) do Σt
            # Σ>(t) –> ΣR(ω) & Σ<(t) –> Σ<(ω)
            ΣR = mapslices(x -> ft(ts, x .* Θt; mode = +1)[2], Σt[:,1:2]; dims = 1)
            ΣL = mapslices(x -> ft(ts, x; mode = +1)[2], Σt[:,3:4]; dims = 1)   

            GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)

            # GR(ω) –> G>(t) & G<(ω) –> G<(t)
            GG = mapslices(x -> ft(ωs, im / 1π * imag(x); mode = -1)[2], GR; dims=1)
            GL = mapslices(x -> ft(ωs, im / 2π * imag(x); mode = -1)[2], GL; dims=1)

            Σt′ = calculate_Σt(model, ρcG, ρcL, GγG, GγL, GG, GL)
            
            return Σt′
        end

        ΣR = mapslices(x -> ft(ts, x .* Θt; mode = +1)[2], Σt[:,1:2]; dims = 1)
        ΣL = mapslices(x -> ft(ts, x; mode = +1)[2], Σt[:,3:4]; dims = 1)
        
        GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)
        
        GG = mapslices(x -> ft(ωs, im / 1π * imag(x); mode = -1)[2], GR; dims = 1)
        GL = mapslices(x -> ft(ωs, im / 2π * imag(x); mode = -1)[2], GL; dims = 1)
        
        GdR = ft(ts, (@. +im * (GG[:,2] * ~GL[:,1] - GL[:,2] * ~GG[:,1]) * Θt); mode=+1)[2]

        TcR = abs2(model.V₀) * GdR
        ρ0cR = ft(ts, ρc .* Θt; mode=+1)[2]

        GG, GL, GdR, ρ0cR + ρ0cR .* TcR .* ρ0cR
    end

    # Kondo temperature estimation
    ρω0 = -1/(2π) * imag(integrate(ts, ρcG - ρcL))
    ωK = exp(model.ε₀f / (2 * abs2(model.V₀) * ρω0))

    # c-electrons
    if !dmft
        GᶜL = let
            itp = extrapolate(interpolate((ts,), ρcL, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        GᶜG = let
            itp = extrapolate(interpolate((ts,), ρcG, Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        dos_c = nothing
    elseif timematrix
        @assert false "unsupported"
    else
        # Non-interacting conduction-electron GF
        GᶜL = [GreenFunction(+im * nf(+ε, β) * ones(1, 1), SkewHermitian) for ε in εs]
        GᶜG = [GreenFunction(-im * nf(-ε, β) * ones(1, 1), SkewHermitian) for ε in εs]

        dos_c = gaussian.(εs, 0.0, 1.0)
    end

    # gaussian pulse
    (; ω₀, Ω₀) = model

    GᵞL₀(t, t′) = -im * (0 + Nᵧ * sech((0.5(t + t′) - T₀) * Ω₀)^2) * exp(-1im * ω₀ * (t - t′)) * sech((t - t′) * Ω₀)^2
    GᵞG₀(t, t′) = -im * (1 + Nᵧ * sech((0.5(t + t′) - T₀) * Ω₀)^2) * exp(-1im * ω₀ * (t - t′)) * sech((t - t′) * Ω₀)^2

    GγL = GreenFunction(GᵞL₀(0.0, 0.0) * ones(1, 1), SkewHermitian)
    GγG = GreenFunction(GᵞG₀(0.0, 0.0) * ones(1, 1), SkewHermitian)

    # auxiliary-particles
    if timematrix
        # Trim the time-grid to contain 2 Kondo time-scale. Looks like 4 timescales but G∞ will "chop" it in half
        τK = inv(ωK)
        r = argmin(abs.(ts .+ 2τK)):argmin(abs.(ts .- 2τK))

        GfL = GreenFunction(G∞(GL[r, 2]), SkewHermitian)
        GfG = GreenFunction(G∞(GG[r, 2]), SkewHermitian)

        GbL = GreenFunction(G∞(GL[r, 1]), SkewHermitian)
        GbG = GreenFunction(G∞(GG[r, 1]), SkewHermitian)

        # initial time-grid (equidistant)
        t0 = ts[1:size(GfL, 1)] .- ts[size(GfL, 1)]
    else
        # find τ = 0 index (corresponds to ∫dω)
        i0 = length(ts) ÷ 2 + 1

        GfL = GreenFunction(GL[i0, 2] * ones(1, 1), SkewHermitian)
        GfG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GbL = GreenFunction(GL[i0, 1] * ones(1, 1), SkewHermitian)
        GbG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        # initial time-grid
        t0 = [0.0]
    end

    # sanity checks
    @assert all(imag(diag(GbL.data)) .<= 0.0)
    @assert all(imag(diag(GfL.data)) .>= 0.0)
    @assert all(-imag(diag(GbL.data)) + 2imag(diag(GfL.data)) .≈ 1.0)
    @assert all(isapprox.(diag(GfG.data), -im; rtol=1e-3))
    @assert all(isapprox.(diag(GbG.data), -im; rtol=1e-3))

    data = TimeData(
        GᶜL,
        GᶜG,
        dos_c,
        εs,
        GfL,
        GfG,
        GbL,
        GbG,
        GγL,
        GγG,
        GᵞL₀,
        GᵞG₀,
        GreenFunction(zero(GfL.data), SkewHermitian),
        GreenFunction(zero(GfG.data), SkewHermitian),
        GreenFunction(zero(GbL.data), SkewHermitian),
        GreenFunction(zero(GbG.data), SkewHermitian),
        GreenFunction(zero(GbL.data), SkewHermitian),
        GreenFunction(zero(GbG.data), SkewHermitian),
        GreenFunction(zero(GγL.data), SkewHermitian),
        GreenFunction(zero(GγG.data), SkewHermitian),
    )

    for t in eachindex(t0), t′ in eachindex(t0)
        Σ!(t0, t, t′, model, data)
    end

    return (Gd, Gc, ωK, data, t0)
end
