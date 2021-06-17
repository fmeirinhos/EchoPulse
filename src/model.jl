"""
    TimeData

Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
  GᶜL::Any
  GᶜG::Any
  ρᶜ::Union{Nothing,Vector{Float64}}
  ωᶜ::Union{Nothing,Vector{Float64}}

  GᶠL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᶠG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᵇL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᵇG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᵞL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᵞG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  GᵞL₀::Function
  GᵞG₀::Function

  ΣᶠL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  ΣᶠG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  ΣᵇL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  ΣᵇG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  ΣᵞL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  ΣᵞG::GreenFunction{T,2,Array{T,2},SkewHermitian}
  TᶜL::GreenFunction{T,2,Array{T,2},SkewHermitian}
  TᶜG::GreenFunction{T,2,Array{T,2},SkewHermitian}
end

function TimeData(model::PhotonAssistedSIAM, nca::FreqNCA, imG; timematrix=false, ωᶜ=nothing, T₀=nothing, Nᵧ=nothing)
  @unpack mesh, data = nca

  # Can compute ∑ₖ Gᶜₖ(τ)
  if isnothing(ωᶜ)
    GᶜL = let
      L = ft(mesh, im / (2π) * data.ρc .* data.fermi; mode=-1)
      itp = extrapolate(interpolate((L[1],), +L[2], Gridded(Linear())), 0)
      (t, t′) -> itp(t - t′)
    end
    GᶜG = let
      G = ft(mesh, im / (2π) * data.ρc .* data.fermi⁻; mode=-1)
      itp = extrapolate(interpolate((G[1],), -G[2], Gridded(Linear())), 0)
      (t, t′) -> itp(t - t′)
    end
    ρᶜ = nothing

    # Consider many c-electron states
  elseif timematrix
    ts = ft(mesh, zero.(mesh); mode=-1)[1]

    GᶜL = [GreenFunction(G∞(+1.0im * NCA.nf(+ω, nca.β) .* exp.(-1.0im * ω * ts)), SkewHermitian) for ω in ωᶜ]
    GᶜG = [GreenFunction(G∞(-1.0im * NCA.nf(-ω, nca.β) .* exp.(-1.0im * ω * ts)), SkewHermitian) for ω in ωᶜ]
    ρᶜ = NCA.gaussian.(ωᶜ, 0.0, model.σ) .* (ωᶜ[2] - ωᶜ[1])
  else
    GᶜL = [GreenFunction(+1.0im * ones(1, 1) * NCA.nf(+ω, nca.β), SkewHermitian) for ω in ωᶜ]
    GᶜG = [GreenFunction(-1.0im * ones(1, 1) * NCA.nf(-ω, nca.β), SkewHermitian) for ω in ωᶜ]
    ρᶜ = NCA.gaussian.(ωᶜ, 0.0, model.σ) .* (ωᶜ[2] - ωᶜ[1])
  end

  # gaussian pulse
  @unpack ω₀, Ω₀ = model

  # pulse shape s.t. ∫dt ξ(t)^2 = 1
  ξ(t) = (Ω₀^2 / 2π)^(1 / 4) * exp(-t^2 * Ω₀^2 / 4)

  n = NCA.integrate(mesh, data.ργ .* data.bose) / (2π)
  N̄ᵧ(T) = isnothing(Nᵧ) ? n : Nᵧ * √(2π / Ω₀^2) * ξ(T - T₀)^2

  GᵞL₀(t, t′) = -im * ξ(t - t′)^2 * exp(-1im * ω₀ * (t - t′)) * √(2π / Ω₀^2) * (0 + N̄ᵧ(0.5(t + t′)))
  GᵞG₀(t, t′) = -im * ξ(t - t′)^2 * exp(-1im * ω₀ * (t - t′)) * √(2π / Ω₀^2) * (1 + N̄ᵧ(0.5(t + t′)))

  GᵞL = GreenFunction(GᵞL₀(0.0, 0.0) * ones(1, 1), SkewHermitian)
  GᵞG = GreenFunction(GᵞG₀(0.0, 0.0) * ones(1, 1), SkewHermitian)

  # slave-particles
  Q = NCA.integrate(mesh, -imG[:, 1, 2] + 2imG[:, 2, 2]) / 2π

  if timematrix
    ts = ft(mesh, zero.(mesh); mode=-1)[1]

    # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
    L = mapslices(x -> ft(mesh, 1im / 2π * x; mode=-1)[2], imG[:, :, 2] / Q; dims=1)
    G = mapslices(x -> ft(mesh, 1im / 2π * x; mode=-1)[2], imG[:, :, 1]; dims=1)

    GᶠL = GreenFunction(G∞(L[:, 2]), SkewHermitian)
    GᶠG = GreenFunction(G∞(G[:, 2]), SkewHermitian)

    GᵇL = GreenFunction(G∞(L[:, 1]), SkewHermitian)
    GᵇG = GreenFunction(G∞(G[:, 1]), SkewHermitian)

    t0 = ts[1:(length(ts) ÷ 2)]
    @assert ts[1:2:end] ≈ t0 .- reverse(t0)
    @assert size(t0, 1) == size(GᶠL, 1)
  else
    ∫dω(f) = integrate(mesh.mesh, f) / (2π)

    GᶠL = GreenFunction(im * ones(1, 1) * ∫dω(imG[:, 2, 2]) / Q, SkewHermitian)
    GᶠG = GreenFunction(-im * ones(1, 1), SkewHermitian)

    GᵇL = GreenFunction(im * ones(1, 1) * ∫dω(imG[:, 1, 2]) / Q, SkewHermitian)
    GᵇG = GreenFunction(-im * ones(1, 1), SkewHermitian)

    @assert imag(GᶠL[1, 1]) >= 0.0
    @assert imag(GᵇL[1, 1]) <= 0.0
    @assert 2imag(GᶠL[1, 1]) - imag(GᵇL[1, 1]) ≈ 1.0
    @assert GᶠG[1, 1] == -im
    @assert GᵇG[1, 1] == -im
  end

  data = TimeData(GᶜL, GᶜG, ρᶜ, ωᶜ, GᶠL, GᶠG, GᵇL, GᵇG, GᵞL, GᵞG, GᵞL₀, GᵞG₀, GreenFunction(zero(GᶠL.data), SkewHermitian),
                  GreenFunction(zero(GᶠG.data), SkewHermitian), GreenFunction(zero(GᵇL.data), SkewHermitian),
                  GreenFunction(zero(GᵇG.data), SkewHermitian), GreenFunction(zero(GᵇL.data), SkewHermitian),
                  GreenFunction(zero(GᵇG.data), SkewHermitian), GreenFunction(zero(GᵇL.data), SkewHermitian),
                  GreenFunction(zero(GᵇG.data), SkewHermitian))

  if timematrix
    for t in eachindex(t0), t′ in eachindex(t0)
      Σ_t(model, data, t0, t, t′)
    end

    return (data, t0)
  else
    return (data, 0.0)
  end
end
