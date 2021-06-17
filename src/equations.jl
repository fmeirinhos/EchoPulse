function f_vt(model::PhotonAssistedSIAM, data::TimeData, ts, t1, t2, T₀)
  @unpack GᶠL, GᶠG, GᵇL, GᵇG, ΣᶠL, ΣᶠG, ΣᵇL, ΣᵇG, GᵞL, GᵞG, TᶜL, TᶜG, ΣᵞL, ΣᵞG = data

  ∫dt(x...) = integrate(ts, t1, t2, x...)

  # pseudo fermion
  dGᶠL = -1.0im * (model.ε₀f * GᶠL[t1, t2] + ∫dt(1, t1, ΣᶠG, GᶠL) - ∫dt(1, t2, ΣᶠL, GᶠG))
  dGᶠG = -1.0im * (model.ε₀f * GᶠG[t1, t2] + ∫dt(t2, t1, ΣᶠG, GᶠG))

  # slave boson
  dGᵇL = -1.0im * (∫dt(1, t1, ΣᵇG, GᵇL) - ∫dt(1, t2, ΣᵇL, GᵇG))
  dGᵇG = -1.0im * (∫dt(t2, t1, ΣᵇG, GᵇG))

  # laser pulse
  h0 = 0.5im * ((T₀ - 0.5(ts[t1] + ts[t2])) * model.Ω₀^2) - (model.ω₀ + 1.0im * (ts[t1] - ts[t2]) * model.Ω₀^2)
  dGᵞL = -1.0im * (h0 * GᵞL[t1, t2])# + ∫dt(1, t1, ΣᵞG, GᵞL) - ∫dt(1, t2, ΣᵞL, GᵞG) - ∫dt(t2, t1, ΣᵞL, GᵞL))
  dGᵞG = -1.0im * (h0 * GᵞL[t1, t2])# + ∫dt(t2, t1, ΣᵞG, GᵞG) - ∫dt(1, t1, ΣᵞL, GᵞG) + ∫dt(1, t2, ΣᵞG, GᵞL))

  out = [dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]

  if isnothing(data.ωᶜ)
    return out
  else
    @unpack GᶜL, GᶜG, TᶜL, TᶜG, ωᶜ = data
    dGᶜL = [-1.0im * ωᶜ[i] * GᶜL[i][t1, t2] for i in eachindex(ωᶜ)]
    dGᶜL .-= 1.0im * (∫dt(1, t1, TᶜG, GᶜL) - ∫dt(1, t1, TᶜL, GᶜL) - ∫dt(1, t2, TᶜL, GᶜG) + ∫dt(1, t2, TᶜL, GᶜL))
    dGᶜG = [-1.0im * ωᶜ[i] * GᶜG[i][t1, t2] for i in eachindex(ωᶜ)]
    dGᶜG .-= 1.0im * (∫dt(1, t1, TᶜG, GᶜG) - ∫dt(1, t1, TᶜL, GᶜG) - ∫dt(1, t2, TᶜG, GᶜG) + ∫dt(1, t2, TᶜG, GᶜL))
    return [out; dGᶜL; dGᶜG]
  end
end

# function kernel_vt(model, data, times, t1, t2, τ)
#   @unpack GᶠL, GᶠG, GᵇL, GᵇG, ΣᶠL, ΣᶠG, ΣᵇL, ΣᵇG = data
#   Θ(i, j) = i >= j # Heaviside function

#   k1 = Θ(t1, τ) * (ΣᶠG[t1, τ] * GᶠL[τ, t2]) - Θ(t2, τ) * (ΣᶠL[t1, τ] * GᶠG[τ, t2])
#   k2 = Θ(τ, t2) * Θ(t1, τ) * (ΣᶠG[t1, τ] * GᶠG[τ, t2])
#   k3 = Θ(t1, τ) * (ΣᵇG[t1, τ] * GᵇL[τ, t2]) - Θ(t2, τ) * (ΣᵇL[t1, τ] * GᵇG[τ, t2])
#   k4 = Θ(τ, t2) * Θ(t1, τ) * (ΣᵇG[t1, τ] * GᵇG[τ, t2])

#   return [k1, k2, k3, k4]
# end

"""
Calculates the self-energies in time
"""
function Σ_t(model::PhotonAssistedSIAM, data::TimeData, ts, t1, t2)
  @unpack GᶜL, GᶜG, GᶠL, GᶠG, GᵇL, GᵇG, GᵞL₀, GᵞG₀, ΣᶠL, ΣᶠG, ΣᵇL, ΣᵇG, ΣᵞL, ΣᵞG, TᶜL, TᶜG, ρᶜ = data

  if (n = size(GᶠL, 2)) > size(ΣᶠL, 2)
    resize!(ΣᶠL, n)
    resize!(ΣᶠG, n)
    resize!(ΣᵇL, n)
    resize!(ΣᵇG, n)
    resize!(ΣᵞL, n)
    resize!(ΣᵞG, n)
    resize!(TᶜL, n)
    resize!(TᶜG, n)
  end

  if isnothing(ρᶜ)
    GᶜL_ = GᶜL(ts[t1], ts[t2])
    GᶜG_ = GᶜG(ts[t1], ts[t2])
  else
    GᶜL_ = sum(GᶜL[i][t1, t2] * ρᶜ[i] for i in eachindex(ρᶜ))
    GᶜG_ = sum(GᶜG[i][t1, t2] * ρᶜ[i] for i in eachindex(ρᶜ))
  end

  V²L = model.V₀^2 + im * model.Γ₀^2 * (GᵞL₀(ts[t1], ts[t2]) + GᵞG₀(ts[t2], ts[t1]))
  V²G = model.V₀^2 + im * model.Γ₀^2 * (GᵞG₀(ts[t1], ts[t2]) + GᵞL₀(ts[t2], ts[t1]))

  ΣᶠL[t1, t2] = +1.0im * V²L * GᵇL[t1, t2] * GᶜL_
  ΣᶠG[t1, t2] = +1.0im * V²G * GᵇG[t1, t2] * GᶜG_
  ΣᵇL[t1, t2] = -2.0im * V²L * GᶠL[t1, t2] * (-GᶜG_')
  ΣᵇG[t1, t2] = -2.0im * V²G * GᶠG[t1, t2] * (-GᶜL_')

  # ΣᵞL[t1, t2] = +2.0 * model.Γ₀^2 * (GᶜG_ * GᶠL[t1, t2] * GᵇG[t2, t1] + GᶜL_ * GᶠG[t1, t2] * GᵇL[t2, t1])
  # ΣᵞG[t1, t2] = +2.0 * model.Γ₀^2 * (GᶜL_ * GᶠG[t1, t2] * GᵇL[t2, t1] + GᶜG_ * GᶠL[t1, t2] * GᵇG[t2, t1])
  ΣᵞL[t1, t2] = -2.0im * model.Γ₀^2 * (GᶠL[t1, t2] * GᵇG[t2, t1] + GᶠG[t1, t2] * GᵇL[t2, t1])
  ΣᵞG[t1, t2] = -2.0im * model.Γ₀^2 * (GᶠG[t1, t2] * GᵇL[t2, t1] + GᶠL[t1, t2] * GᵇG[t2, t1])

  TᶜL[t1, t2] = +1.0im * V²L * GᶠL[t1, t2] * GᵇG[t2, t1]
  TᶜG[t1, t2] = +1.0im * V²G * GᶠG[t1, t2] * GᵇL[t2, t1]
  return
end
