"""
    FrequencyData

Data needed to solve the Kadanoff-Baym equation in the imaginary-time contour
"""
struct FrequencyData
  nca::FreqNCA
  imG::Array{Float64,3}
  Gᵈ::Array{ComplexF64,1}
  ωK::Float64                     # Kondo frequency
end

function FrequencyData(model::PhotonAssistedSIAM, mesh::AbstractArray, β::Float64)  
  # Equilibrium model
  nca = let
    mesh = LinearMesh(first(mesh), last(mesh), length(mesh))

    ρc = 2π * NCA.gaussian.(mesh, 0.0, 1.0)
    ργ = 2π * NCA.gaussian.(mesh, +model.ω₀, model.Ω₀)

    data = PhotonAssistedSIAMData(mesh, β, ρc, ργ)#, ω->NCA.nf(ω,β), ω->NCA.nf(-ω,β))

    FreqNCA(model=model, mesh=mesh, data=data, β=β)
  end

  # Obtain pseudo-particle Greater and Lesser functions
  imG = let
    x0 = cat(
      abs2(nca.model.V₀) * hcat(-nca.data.ρc, -nca.data.ρc),
      abs2(nca.model.V₀) * hcat(-nca.data.ρc, +nca.data.ρc),
      dims=3) # initial guesses

    # Solve NCA equations
    imΣ = nlsolve(x0, method=:anderson, beta=0.9, show_trace=:true) do imΣ′
      imG = NCA.slave_particle_imG(nca, imΣ′)
      imΣ = NCA.slave_particle_imΣ(nca, imG)
      imΣ - imΣ′
    end.zero

    NCA.slave_particle_imG(nca, imΣ)
  end

  return FrequencyData(
    nca, 
    imG, 
    NCA.projector(nca, imG)[2], 
    NCA.kondo_temp(nca.mesh, nca.data.ρc ./ (2π), model.ε₀f, model.V₀)
    )
end

"""
    TimeData

Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
  GᶜL#::Union{Any, GreenFunction{T, Array{T,2}, Lesser}}
  GᶜG#::Union{Any, GreenFunction{T, Array{T,2}, Greater}}
  ρᶜ::Union{Nothing, Vector{Float64}}
  ωᶜ::Union{Nothing, Vector{Float64}}

  GᶠL::GreenFunction{T, Array{T,2}, Lesser}
  GᶠG::GreenFunction{T, Array{T,2}, Greater}
  GᵇL::GreenFunction{T, Array{T,2}, Lesser}
  GᵇG::GreenFunction{T, Array{T,2}, Greater}
  GᵞL₀::Function
  GᵞG₀::Function
  
  ΣᶠL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᶠG::GreenFunction{T, Array{T,2}, Greater}
  ΣᵇL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᵇG::GreenFunction{T, Array{T,2}, Greater}
  TᶜL::GreenFunction{T, Array{T,2}, Lesser}
  TᶜG::GreenFunction{T, Array{T,2}, Greater}
end

function TimeData(model::PhotonAssistedSIAM, freqs::FrequencyData; timematrix=false, ωᶜ=nothing, T₀=nothing, Nᵧ=nothing)
  @unpack nca, imG = freqs
  @unpack mesh, data = nca

  # Can compute ∑ₖ Gᶜₖ(τ)
  if isnothing(ωᶜ)
    GᶜL = let
      L = ft(mesh, im / (2π) * data.ρc .* data.fermi, mode=-1)
      extrapolate(interpolate((L[1],), +L[2], Gridded(Linear())), 0)
    end
    GᶜG = let
      G = ft(mesh, im / (2π) * data.ρc .* data.fermi⁻, mode=-1)
      extrapolate(interpolate((G[1],), -G[2], Gridded(Linear())), 0)
    end
    ρᶜ = nothing

  # Consider many c-electron states
  elseif timematrix
    ts = ft(mesh, zero.(mesh); mode=-1)[1]

    GᶜL = [GreenFunction(G∞(+1.0im * NCA.nf(+ω, nca.β) .* exp.(-1.0im * ω * ts)), Lesser) for ω in ωᶜ]
    GᶜG = [GreenFunction(G∞(-1.0im * NCA.nf(-ω, nca.β) .* exp.(-1.0im * ω * ts)), Greater) for ω in ωᶜ]
    ρᶜ = NCA.gaussian.(ωᶜ, 0.0, model.σ) .* (ωᶜ[2] - ωᶜ[1])
  else
    GᶜL = [GreenFunction(+1.0im * ones(1,1) * NCA.nf(+ω, nca.β), Lesser) for ω in ωᶜ]
    GᶜG = [GreenFunction(-1.0im * ones(1,1) * NCA.nf(-ω, nca.β), Greater) for ω in ωᶜ]
    ρᶜ = NCA.gaussian.(ωᶜ, 0.0, model.σ) .* (ωᶜ[2] - ωᶜ[1])
  end

  # gaussian pulse
  @unpack ω₀, Ω₀ = model

  # pulse shape s.t. ∫dt ξ(t)^2 = 1
  ξ(t) = (Ω₀^2 / 2π)^(1/4) * exp(-t^2 * Ω₀^2 / 4)

  n = NCA.integrate(nca.mesh, data.ργ .* data.bose) / (2π)
  N̄ᵧ(T) = isnothing(Nᵧ) ? n : Nᵧ * √(2π/Ω₀^2) * ξ(T-T₀)^2

  GᵞL₀(T,τ) = -im * ξ(τ)^2 * exp(-1im * ω₀ * τ) * √(2π/Ω₀^2) * (0 + N̄ᵧ(T))
  GᵞG₀(T,τ) = -im * ξ(τ)^2 * exp(-1im * ω₀ * τ) * √(2π/Ω₀^2) * (1 + N̄ᵧ(T))

  # slave-particles
  if timematrix
    # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
    L = mapslices(x -> ft(mesh, 1im/2π * x; mode=-1)[2], imG[:,:,2], dims=1)
    G = mapslices(x -> ft(mesh, 1im/2π * x; mode=-1)[2], imG[:,:,1], dims=1)

    GᶠL = GreenFunction(G∞(L[:,2]), Lesser)
    GᶠG = GreenFunction(G∞(G[:,2]), Greater)

    GᵇL = GreenFunction(G∞(L[:,1]), Lesser)  
    GᵇG = GreenFunction(G∞(G[:,1]), Greater)

    t0 = ts[1:length(ts)÷2]
    @assert ts[1:2:end] ≈ t0 .- reverse(t0)
    @assert size(t0, 1) == size(GᶠL, 1)
  else
    ∫dω(f) = integrate(mesh.mesh, f) / (2π)

    GᶠL = GreenFunction(im * ones(1,1) * ∫dω(imG[:,2,2]), Lesser)
    GᶠG = GreenFunction(-im * ones(1,1), Greater)

    GᵇL = GreenFunction(im * ones(1,1) * ∫dω(imG[:,1,2]), Lesser) 
    GᵇG = GreenFunction(-im * ones(1,1), Greater)

    @assert imag(GᶠL[1,1]) >= 0.0
    @assert imag(GᵇL[1,1]) <= 0.0
    @assert 2imag(GᶠL[1,1]) - imag(GᵇL[1,1]) ≈ 1.0

  end

  data = TimeData(GᶜL,GᶜG,ρᶜ,ωᶜ,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,
      GreenFunction(zero(GᶠL.data), Lesser),
      GreenFunction(zero(GᶠG.data), Greater),
      GreenFunction(zero(GᵇL.data), Lesser),
      GreenFunction(zero(GᵇG.data), Greater),
      GreenFunction(zero(GᵇL.data), Lesser),
      GreenFunction(zero(GᵇG.data), Greater))

  if timematrix
    for t in eachindex(t0), t′ in eachindex(t0)
      Σ_t(model, data, t0, t, t′)
    end

    return (data, t0)
  else
    return (data, 0.0)
  end
end
