"""
Photon assisted single impurity Anderson Model
"""
struct PhotonAssistedSIAM{T<:AbstractFloat}
  # physical parameters
  V₀::T     # Hybridization strength
  ε₀f::T    # F-electron ground state e
  μ::T      # Chemical potential

  ω₀::T     # Photon frequency
  Ω₀::T     # Photon width
  Γ₀::T     # Photon hybridization strength

  function PhotonAssistedSIAM(; V₀, ε₀f, ω₀, Ω₀, Γ₀, μ=0.0)
    t = promote(V₀, ε₀f, μ, ω₀, Ω₀, Γ₀)
    return new{eltype(t)}(t...)
  end
end

"""
    TimeData

Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
    GcL::Any
    GcG::Any
    dos_c::Union{Nothing,Vector{Float64}}
    ωc::Vector{Float64}

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

"""
"""
function TimeData(; model::PhotonAssistedSIAM, ωs, β, T₀, Nᵧ, ωc, dmft, timematrix)
    # Calculate ρc>(t) and ρc<(t) from ρc(ω) = ∑k Gc(k; ω)
    ρcL, ρcG = let
        # c-electron density of states
        dos_c(ω) = 2π * gaussian(ω, 0.0, 1.0)

        η = 0.05 # Artificial damping solves a weird behaviour at large `τ` after summing all frequencies
        ρc = -1/π * integrate(ωc, dos_c.(ωc) .* (ε -> -η ./ ((ωs .- ε).^2 .+ η^2)).(ωc))

        ρcL = ft(ωs, +im / (2π) * ρc .* nf.(+ωs, β); mode = -1)[2]
        ρcG = ft(ωs, -im / (2π) * ρc .* nf.(-ωs, β); mode = -1)[2]

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
        Σt, success = fixed_point(hcat(ρc, ρc, ρc, -ρc)) do Σt
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

        @assert success "Failed to converge"

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
        GᶜL = [GreenFunction(+im * ones(1, 1) * nf(+ω, nca.β), SkewHermitian) for ω in ωc]
        GᶜG = [GreenFunction(-im * ones(1, 1) * nf(-ω, nca.β), SkewHermitian) for ω in ωc]

        dos_c = (ωc[2] - ωc[1]) .* gaussian.(ωc, 0.0, 1.0)
    end

    # gaussian pulse
    @unpack ω₀, Ω₀ = model

    N̄ᵧ(T) = Nᵧ * √(2π / Ω₀^2) * gaussian(T, T₀, inv(Ω₀))
    GᵞL₀(t, t′) = -im * (0 + N̄ᵧ(0.5(t + t′))) * exp(-1im * ω₀ * (t - t′)) * gaussian(t - t′, 0, inv(Ω₀)) * √(2π / Ω₀^2)
    GᵞG₀(t, t′) = -im * (1 + N̄ᵧ(0.5(t + t′))) * exp(-1im * ω₀ * (t - t′)) * gaussian(t - t′, 0, inv(Ω₀)) * √(2π / Ω₀^2)

    GγL = GreenFunction(GᵞL₀(0.0, 0.0) * ones(1, 1), SkewHermitian)
    GγG = GreenFunction(GᵞG₀(0.0, 0.0) * ones(1, 1), SkewHermitian)

    # slave-particles
    if timematrix
        # Trim the time-grid to contain 2 Kondo time-scale. Looks like 4 timescales but G∞ will "chop" it in half
        τK = inv(ωK)
        r = argmin(abs.(ts .+ 2τK)):argmin(abs.(ts .- 2τK))

        GᶠL = GreenFunction(G∞(GL[r, 2]), SkewHermitian)
        GᶠG = GreenFunction(G∞(GG[r, 2]), SkewHermitian)

        GᵇL = GreenFunction(G∞(GL[r, 1]), SkewHermitian)
        GᵇG = GreenFunction(G∞(GG[r, 1]), SkewHermitian)

        # initial time-grid
        L = size(GᶠL, 1)
        t0 = ts[1:L] .- ts[L] # (equidistant)
    else
        i0 = length(ts) ÷ 2 + 1
        GᶠL = GreenFunction(GL[i0, 2] * ones(1, 1), SkewHermitian)
        GᶠG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GᵇL = GreenFunction(GL[i0, 1] * ones(1, 1), SkewHermitian)
        GᵇG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        @assert imag(GᵇL[1, 1]) <= 0.0
        @assert imag(GᶠL[1, 1]) >= 0.0
        @assert -imag(GᵇL[1, 1]) + 2imag(GᶠL[1, 1]) ≈ 1.0
        @assert GᶠG[1, 1] == -im
        @assert GᵇG[1, 1] == -im

        # initial time-grid
        t0 = [0.0]
    end

    data = TimeData(
        GᶜL,
        GᶜG,
        dos_c,
        ωc,
        GᶠL,
        GᶠG,
        GᵇL,
        GᵇG,
        GγL,
        GγG,
        GᵞL₀,
        GᵞG₀,
        GreenFunction(zero(GᶠL.data), SkewHermitian),
        GreenFunction(zero(GᶠG.data), SkewHermitian),
        GreenFunction(zero(GᵇL.data), SkewHermitian),
        GreenFunction(zero(GᵇG.data), SkewHermitian),
        GreenFunction(zero(GᵇL.data), SkewHermitian),
        GreenFunction(zero(GᵇG.data), SkewHermitian),
        GreenFunction(zero(GγL.data), SkewHermitian),
        GreenFunction(zero(GγG.data), SkewHermitian),
    )

    for t in eachindex(t0), t′ in eachindex(t0)
        Σ!(t0, t, t′, model, data)
    end

    return (Gd, Gc, ωK, data, t0)
end
