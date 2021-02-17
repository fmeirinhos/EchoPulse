"""
Parameters for the photon-assisted SIAM / PAM
"""
Base.@kwdef mutable struct PhotonAssistedModel
  # single impurity Anderson model parameters
  V₀::Float64     # hybridization strength
  ε₀f::Float64    # f-electron ground state energy
  λ::Float64=NaN # gauge chemical potential
  σ::Float64      # c-electron gaussian bandwidth

  # pulse parameters
  Γ₀::Float64     # light-matter coupling strength
  ω₀::Float64     # light pulse energy
  Ω::Float64      # light pulse bandwidth
  tc::Float64     # light pulse peak time
  Π::Float64      # light pulse max occupation
end

"""
Data needed to solve the Kadanoff-Baym equation in the imaginary-time contour
"""
struct FrequencyData
  nca::FreqNCA
  imG_t::Array{Float64,2}
  G_d::Array{ComplexF64,1}
  ωK::Float64                     # Kondo frequency
end

function FrequencyData(model::PhotonAssistedModel, mesh::AbstractArray, β::Float64)
  # Equilibrium model
  nca = FreqNCA(
    model=SIAM(
      V₀=model.V₀,
      ε₀f=model.ε₀f,
      Nc₀=x-> 2pi * NCA.gaussian(x, -0.0, model.σ)),
    β=β,
    mesh=LinearMesh(first(mesh), last(mesh), length(mesh)))

  imΣ_t = let
    # Initial guesses
    x0 = abs2(nca.model.V₀) * hcat(nca.data.Nc, nca.data.Nc)

    # Find fixed-point
    nlsolve(x0, method=:anderson, beta=0.9, show_trace=:true) do imΣ
      NCA.calculate_λ!(nca, imΣ)
      NCA.slave_particle_imΣ!(nca, NCA.slave_particle_imG!(nca, imΣ)) - imΣ
    end.zero
  end

  imG_t = NCA.slave_particle_imG!(nca, imΣ_t)
  G_d = NCA.projector(nca, imG_t)[2]

  ωK = NCA.kondo_temp(nca.mesh, nca.data.Nc ./ (2pi), model.ε₀f, model.V₀)

  return FrequencyData(nca, imG_t, G_d, ωK)
end

"""
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

function TimeData(model::PhotonAssistedModel, freqs::FrequencyData; timematrix=false, ωᶜ=nothing)
  @unpack nca, imG_t = freqs
  @unpack mesh, data = nca

  if !isnothing(ωᶜ)
    GᶜL = [GreenFunction(+im * ones(1,1) .* NCA.nf.(+ω, nca.β), Lesser) for ω in ωᶜ]
    GᶜG = [GreenFunction(-im * ones(1,1) .* NCA.nf.(-ω, nca.β), Greater) for ω in ωᶜ]
    ρᶜ = nca.model.Nc₀.(ωᶜ) .* (ωᶜ[2] - ωᶜ[1])/(2pi)
  else
    # conduction electrons
    ts, L_ = ft(mesh, im/(2pi) * data.Nc .* data.fermi, mode=-1)
    GᶜL = extrapolate(interpolate((ts,), +L_, Gridded(Linear())), 0)
    ts, G_ = ft(mesh, im/(2pi) * data.Nc .* data.fermi⁻, mode=-1)
    GᶜG = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)
    ρᶜ = nothing
  end

  # thz pulse (shape s.t. ∫dt ξ(t)^2 = 1)
  @unpack ω₀, Ω, tc, Π = model
  ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
  n_ξ(t) = (Π / Ω * sqrt(2pi)) * ξ(t)^2
  GᵞL₀(T,τ) = -1im * (n_ξ(T) + 0) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω^2 / 4)/2)
  GᵞG₀(T,τ) = -1im * (n_ξ(T) + 1) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω^2 / 4)/2)

  # slave-particles
  if timematrix  
    # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
    L = mapslices(x -> ft(mesh, (1/pi)im * x .* data.fermi; mode=-1)[2], imG_t, dims=1)
    G = mapslices(x -> ft(mesh, (1/pi)im * x .* data.fermi⁻; mode=-1)[2], imG_t, dims=1)

    GᶠL = GreenFunction(G∞(+L[:,2]), Lesser)
    GᶠG = GreenFunction(G∞(-G[:,2]), Greater)

    GᵇL = GreenFunction(G∞(-L[:,1]), Lesser)  
    GᵇG = GreenFunction(G∞(-G[:,1]), Greater)

    t0 = ts[1:length(ts)÷2]
    @assert ts[1:2:end] ≈ t0 .- reverse(t0)
    @assert size(t0, 1) == size(GᶠL, 1)
  else
    ∫dω(f) = integrate(mesh.mesh, f) / (2π)

    GᶠL = GreenFunction(+im * ones(1,1) * ∫dω(data.fermi .* 2imG_t[:,2]), Lesser)
    GᶠG = GreenFunction(-im * ones(1,1), Greater)

    GᵇL = GreenFunction(-im * ones(1,1) * ∫dω(data.fermi .* 2imG_t[:,1]), Lesser) 
    GᵇG = GreenFunction(-im * ones(1,1), Greater)
  end

  # Update auxiliary chemical potential
  model.λ = nca.model.λ

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
