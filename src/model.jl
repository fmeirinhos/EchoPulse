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
  GᶜL#::GreenFunction{T, Array{T,2}, Lesser}
  GᶜG#::GreenFunction{T, Array{T,2}, Greater}
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
end

function TimeData(model::PhotonAssistedModel, freqs::FrequencyData; timematrix=false)
  @unpack nca, imG_t = freqs

  ∫dω(f) = integrate(nca.mesh.mesh, f) / (2π)
  
  # conduction electrons
  ts, L_ = ft(nca.mesh, im/(2pi) * nca.data.Nc .* nca.data.fermi, mode=-1)
  GᶜL = extrapolate(interpolate((ts,), +L_, Gridded(Linear())), 0)
  ts, G_ = ft(nca.mesh, im/(2pi) * nca.data.Nc .* nca.data.fermi⁻, mode=-1)
  GᶜG = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)

  # thz pulse
  @unpack ω₀, Ω, tc, Π = model
  # Pulse shape s.t. ∫dt ξ(t)^2 = 1
  ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
  n_ξ(t) = (Π / Ω * sqrt(2pi)) * ξ(t)^2
  GᵞL₀(T,τ) = -1im * (n_ξ(T) + 0) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω^2 / 4)/2)
  GᵞG₀(T,τ) = -1im * (n_ξ(T) + 1) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω^2 / 4)/2)

  # Update auxiliary chemical potential
  model.λ = nca.model.λ

  # slave-particles
  if timematrix  
    # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
    L = mapslices(imG_t, dims=1) do x
      ft(nca.mesh, (1/pi)im * x .* nca.data.fermi; mode=-1)[2]
    end
    G = mapslices(imG_t, dims=1) do x
      ft(nca.mesh, (1/pi)im * x .* nca.data.fermi⁻; mode=-1)[2]
    end

    GᶠL = GreenFunction(G∞(+L[:,2]), Lesser)
    GᶠG = GreenFunction(G∞(-G[:,2]), Greater)

    GᵇL = GreenFunction(G∞(-L[:,1]), Lesser)  
    GᵇG = GreenFunction(G∞(-G[:,1]), Greater)

    # NOTE: double check these boundaries
    t0 = sort(-ts[1+length(ts)÷2:end])
  else
    GᶠL = GreenFunction(+im * ones(1,1) * ∫dω(nca.data.fermi .* 2imG_t[:,2]), Lesser)
    GᶠG = GreenFunction(-im * ones(1,1), Greater)

    GᵇL = GreenFunction(-im * ones(1,1) * ∫dω(nca.data.fermi .* 2imG_t[:,1]), Lesser) 
    GᵇG = GreenFunction(-im * ones(1,1), Greater)
  end

  data = TimeData(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,
      GreenFunction(zero(GᶠL.data), Lesser),
      GreenFunction(zero(GᶠG.data), Greater),
      GreenFunction(zero(GᵇL.data), Lesser),
      GreenFunction(zero(GᵇG.data), Greater))

  if timematrix
    @assert size(t0, 1) == size(GᶠL, 1)
    for t in eachindex(t0), t′ in eachindex(t0)
      Σ_t(model, data, t0, t, t′)
    end

    return (data, t0)
  else
    return (data, 0.0)
  end
end
