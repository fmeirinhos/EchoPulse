"""
Vertical stepping function in time
i ∂ₜ G(t,t′) = ...
"""
function f_vt(model::PhotonAssistedModel,data::TimeData,times,t,t′)
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ = model

  ∫dt(x...) = integrate(times,t,t′,x...)

  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f + λ) * GᶠL[t,t′] + ∫dt(1,t,ΣᶠG,GᶠL) - ∫dt(1,t′,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f + λ) * GᶠG[t,t′] + ∫dt(t′,t,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (λ * GᵇL[t,t′] + ∫dt(1,t,ΣᵇG,GᵇL) - ∫dt(1,t′,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (λ * GᵇG[t,t′] + ∫dt(t′,t,ΣᵇG,GᵇG))

  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end

"""
Diagonal stepping function in time
i (∂ₜ + ∂ₜ′) G(t,t′) = ...
"""
function f_dt(model::PhotonAssistedModel,data::TimeData,times,t)
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data

  ∫dt(x...) = integrate(times,t,t,1,t,x...)
  
  # pseudo fermion
  dGᶠL = -1.0im * (∫dt(ΣᶠG,GᶠL) - ∫dt(ΣᶠL,GᶠG) + ∫dt(GᶠL,ΣᶠG) - ∫dt(GᶠG,ΣᶠL))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (∫dt(ΣᵇG,GᵇL) - ∫dt(ΣᵇL,GᵇG) + ∫dt(GᵇL,ΣᵇG) - ∫dt(GᵇG,ΣᵇL))
  dGᵇG = 0.0
  
  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end

"""
Calculates the self-energies in time
"""
function Σ_t(model::PhotonAssistedModel,data::TimeData,times,t,t′)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack V₀,Γ₀ = model

  T = 1//2 * (times[t] + times[t′])
  τ = times[t] - times[t′]
    
  if (n = size(GᶠL,2)) > size(ΣᶠL,2)
    resize!(ΣᶠL,n); resize!(ΣᶠG,n)
    resize!(ΣᵇL,n); resize!(ΣᵇG,n)
  end

  GᶜL_τ = GᶜL(τ)
  GᶜG_τ = GᶜG(τ)

  V²L = V₀^2 + im * Γ₀^2 * (GᵞL₀(T,+τ) + GᵞG₀(T,-τ))
  V²G = V₀^2 + im * Γ₀^2 * (GᵞG₀(T,+τ) + GᵞL₀(T,-τ))
    
  ΣᶠL[t,t′] = +1.0im * V²L * GᵇL[t,t′] * GᶜL_τ
  ΣᶠG[t,t′] = +1.0im * V²G * GᵇG[t,t′] * GᶜG_τ
  ΣᵇL[t,t′] = -2.0im * V²L * GᶠL[t,t′] * (-GᶜG_τ')
  ΣᵇG[t,t′] = -2.0im * V²G * GᶠG[t,t′] * (-GᶜL_τ')
end
