"""
Vertical stepping function in time
i ∂ₜ G(t,t′) = ...
"""
function f_vt(model::PhotonAssistedSIAM,data::TimeData,times,t,t′)
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ = model

  ∫dt(x...) = integrate(times,t,t′,x...)

  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f + λ) * GᶠL[t,t′] + ∫dt(1,t,ΣᶠG,GᶠL) - ∫dt(1,t′,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f + λ) * GᶠG[t,t′] + ∫dt(t′,t,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (λ * GᵇL[t,t′] + ∫dt(1,t,ΣᵇG,GᵇL) - ∫dt(1,t′,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (λ * GᵇG[t,t′] + ∫dt(t′,t,ΣᵇG,GᵇG))

  out = [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
  
  if isnothing(data.ωᶜ)
    return out
  else
    @unpack GᶜL, GᶜG, TᶜL, TᶜG, ωᶜ = data
    dGᶜL = [-1.0im * ωᶜ[i] * GᶜL[i][t,t′] for i in eachindex(ωᶜ)]
    dGᶜL .-= 1.0im * (∫dt(1,t,TᶜG,GᶜL) - ∫dt(1,t,TᶜL,GᶜL) - ∫dt(1,t′,TᶜL,GᶜG) + ∫dt(1,t′,TᶜL,GᶜL))
    dGᶜG = [-1.0im * ωᶜ[i] * GᶜG[i][t,t′] for i in eachindex(ωᶜ)]
    dGᶜG .-= 1.0im * (∫dt(1,t,TᶜG,GᶜG) - ∫dt(1,t,TᶜL,GᶜG) - ∫dt(1,t′,TᶜG,GᶜG) + ∫dt(1,t′,TᶜG,GᶜL))
    return [out; dGᶜL; dGᶜG]
  end
end

"""
Diagonal stepping function in time
i (∂ₜ + ∂ₜ′) G(t,t′) = ...
"""
function f_dt(model::PhotonAssistedSIAM,data::TimeData,times,t)
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data

  ∫dt(x...) = integrate(times,t,t,1,t,x...)
  
  # pseudo fermion
  dGᶠL = -1.0im * (∫dt(ΣᶠG,GᶠL) - ∫dt(ΣᶠL,GᶠG) + ∫dt(GᶠL,ΣᶠG) - ∫dt(GᶠG,ΣᶠL))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (∫dt(ΣᵇG,GᵇL) - ∫dt(ΣᵇL,GᵇG) + ∫dt(GᵇL,ΣᵇG) - ∫dt(GᵇG,ΣᵇL))
  dGᵇG = 0.0
  
  out = [dGᶠL, dGᶠG, dGᵇL, dGᵇG]

  if isnothing(data.ωᶜ)
    return out
  else
    @unpack GᶜL, GᶜG, TᶜL, TᶜG = data
    I = -1.0im * (∫dt(TᶜG,GᶜL) - ∫dt(TᶜL,GᶜG) + ∫dt(GᶜL,TᶜG) - ∫dt(GᶜG, TᶜL))
    dGᶜL = I
    dGᶜG = I
    return [out; dGᶜL; dGᶜG]
  end
end

"""
Calculates the self-energies in time
"""
function Σ_t(model::PhotonAssistedSIAM,data::TimeData,times,t,t′)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG,TᶜL,TᶜG,ρᶜ = data
  @unpack V₀,Γ₀ = model
    
  if (n = size(GᶠL,2)) > size(ΣᶠL,2)
    resize!(ΣᶠL,n); resize!(ΣᶠG,n)
    resize!(ΣᵇL,n); resize!(ΣᵇG,n)
    resize!(TᶜL,n); resize!(TᶜG,n)
  end

  T = 0.5 * (times[t] + times[t′])
  τ = times[t] - times[t′]

  if isnothing(ρᶜ)
    GᶜL_ = GᶜL(τ)
    GᶜG_ = GᶜG(τ)
  else
    GᶜL_ = sum(GᶜL[i][t,t′] * ρᶜ[i] for i in eachindex(ρᶜ))
    GᶜG_ = sum(GᶜG[i][t,t′] * ρᶜ[i] for i in eachindex(ρᶜ))
  end

  V²L = V₀^2 + im * Γ₀^2 * (GᵞL₀(T,+τ) + GᵞG₀(T,-τ))
  V²G = V₀^2 + im * Γ₀^2 * (GᵞG₀(T,+τ) + GᵞL₀(T,-τ))
    
  ΣᶠL[t,t′] = +1.0im * V²L * GᵇL[t,t′] * GᶜL_
  ΣᶠG[t,t′] = +1.0im * V²G * GᵇG[t,t′] * GᶜG_
  ΣᵇL[t,t′] = -2.0im * V²L * GᶠL[t,t′] * (-GᶜG_')
  ΣᵇG[t,t′] = -2.0im * V²G * GᶠG[t,t′] * (-GᶜL_')

  TᶜL[t,t′] = +1.0im * V²L * GᶠL[t,t′] * GᵇG[t′,t] 
  TᶜG[t,t′] = +1.0im * V²G * GᶠG[t,t′] * GᵇL[t′,t]
end
