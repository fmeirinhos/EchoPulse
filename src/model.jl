"""
    FrequencyData

Data needed to solve the Kadanoff-Baym equation in the imaginary-time contour
"""
struct FrequencyData
  nca::FreqNCA
  imG_t::Array{Float64,2}
  G_d::Array{ComplexF64,1}
  ωK::Float64                     # Kondo frequency
end

function FrequencyData(model::PhotonAssistedSIAM, mesh::AbstractArray, β::Float64)
  # c-electrons density of states
  dos(ω) = 2π * NCA.gaussian(ω, 0.0, 1/3) # half-bandwidth ~1
  
  # Equilibrium model
  nca = let
    mesh = LinearMesh(first(mesh), last(mesh), length(mesh))

    ρc = dos.(mesh)
    ργ = let
      ργ′ = mesh .* NCA.lorentzian.(mesh, +model.ω₀, model.Ω₀)
      (2pi) * ργ′ ./ NCA.integrate(mesh, ργ′)
    end

    data = PhotonAssistedNCAData(model, mesh, β, ρc, ργ)

    FreqNCA(model=model, mesh=mesh, data=data, β=β)
  end

  # Obtain tilde spectral functions
  imG_t = let
    x0 = abs2(nca.model.V₀) * hcat(nca.data.ρc, nca.data.ρc) # initial guesses

    # Solve NCA equations
    imΣ = nlsolve(x0, method=:anderson, beta=0.9, show_trace=:true) do imΣ′
      Σ = NCA.slave_particle_Σ!(nca, imΣ′)
      _ = NCA.calculate_λ!(nca, imΣ′, Σ)
      imG = NCA.slave_particle_imG!(nca, imΣ′, Σ)
      
      # Gc_new = let
      #   Σc = abs2(nca.model.V₀) * NCA.projector(nca, imG)[2]

      #   Gc_new = [quadgk(-pi/2, pi/2) do ε
      #     dos(tan(ε) * imag(Σc[i]) + ωᵢ - real(Σc[i]))
      #   end[1]/(2pi) for (i, ωᵢ) in enumerate(nca.mesh.mesh)]

      #   NCA.kramerskronig(nca.mesh, Gc_new, :real) + 1.0im * Gc_new
      # end
      # nca.data.ρc = 2imag(Gc_new)
      # nca.data.ρc_th½ = 2imag(Gc_new .* nca.data.th½)

      imΣ = NCA.slave_particle_imΣ!(nca, imG)
      imΣ - imΣ′
    end.zero

    NCA.slave_particle_imG!(nca, imΣ)
  end

  return FrequencyData(
    nca, 
    imG_t, 
    NCA.projector(nca, imG_t)[2], 
    NCA.kondo_temp(nca.mesh, nca.data.ρc ./ (2pi), model.ε₀f, model.V₀)
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

function TimeData(model::PhotonAssistedSIAM, freqs::FrequencyData; timematrix=false, ωᶜ=nothing, n_ξ=nothing)
  @unpack nca, imG_t = freqs
  @unpack mesh, data = nca

  # Can compute ∑ₖ Gᶜₖ(τ)
  if isnothing(ωᶜ)
    GᶜL = let
      L = ft(mesh, im / (2pi) * data.ρc .* data.fermi, mode=-1)
      extrapolate(interpolate((L[1],), +L[2], Gridded(Linear())), 0)
    end
    GᶜG = let
      G = ft(mesh, im / (2pi) * data.ρc .* data.fermi⁻, mode=-1)
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

  if isnothing(n_ξ)
    GᵞL₀ = let
      L = ft(mesh, -1im/(2pi) * data.ργ .* NCA.nb.(+mesh.mesh, nca.β); mode=-1)
      L = extrapolate(interpolate((L[1],), L[2], Gridded(Linear())), 0)
      (T, τ) -> L(τ)
    end

    GᵞG₀ = let
      G = ft(mesh, +1im/(2pi) * data.ργ .* NCA.nb.(-mesh.mesh, nca.β); mode=-1)
      G = extrapolate(interpolate((G[1],), G[2], Gridded(Linear())), 0)
      (T, τ) -> G(τ)
    end
  else
    @unpack ω₀, Ω₀ = model
    GᵞL₀(T,τ) = -im * (n_ξ(T) + 0) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω₀^2 / 4)/2)
    GᵞG₀(T,τ) = -im * (n_ξ(T) + 1) * exp(-im * τ * ω₀) * exp(-τ^2 * (Ω₀^2 / 4)/2)
  end

  # slave-particles
  if timematrix  
    # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
    L = mapslices(x -> ft(mesh, 1im/pi * x .* data.fermi; mode=-1)[2], imG_t, dims=1)
    G = mapslices(x -> ft(mesh, 1im/pi * x .* data.fermi⁻; mode=-1)[2], imG_t, dims=1)

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
