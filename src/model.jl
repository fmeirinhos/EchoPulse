using Parameters
using NCA
using KadanoffBaym
using NumericalIntegration
using ElasticArrays

@with_kw mutable struct PhotonAssitedModel
  # single impurity Anderson model parameters
  V₀::Float64     # hybridization strength
  ε₀f::Float64    # f-electron ground state energy
  λ₀::Float64     # gauge chemical potential
  σ::Float64      # c-electron lorenztian bandwidth

  # pulse parameters
  Γ₀::Float64     # light-matter coupling strength
  ω₀::Float64     # photon ground state energy
  δ₀::Float64     # photon artificial width

  tc::Float64     # peak arrival time
  Ω::Float64      # frequency bandwidth of the pulse
  ξ=nothing       # pulse shape

  Γ::Float64      # pump and loss strength
  λ=nothing       # loss function
  γ=nothing       # pump function

  hᵧ=nothing      # photon field Hamiltonian
end

struct ProblemData
  GᶜL::GreenFunction
  GᶜG::GreenFunction
  GᶠL::GreenFunction
  GᶠG::GreenFunction
  GᵇL::GreenFunction
  GᵇG::GreenFunction
  GᵞL::GreenFunction
  GᵞG::GreenFunction
  
  Gᶜ˥m::GreenFunction
  Gᶜ˥p::GreenFunction
  Gᶠ˥m::GreenFunction
  Gᶠ˥p::GreenFunction
  Gᵇ˥m::GreenFunction
  Gᵇ˥p::GreenFunction
  
  ΣᶠL::GreenFunction
  ΣᶠG::GreenFunction
  ΣᵇL::GreenFunction
  ΣᵇG::GreenFunction
  
  Σᶠ˥p::GreenFunction
  Σᶠ˥m::GreenFunction
  Σᵇ˥p::GreenFunction
  Σᵇ˥m::GreenFunction
  
  freqs::NCA.AbstractMesh
  fermi::Vector
  bose::Vector
  boltz::Vector

  function ProblemData(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    Gᶜ˥m,Gᶜ˥p,Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,freqs,fermi,bose,boltz)

  new(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    Gᶜ˥m,Gᶜ˥p,Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    typeof(Gᶠ˥m)(ElasticArray(zero(Gᶠ˥m.data))),
    typeof(Gᶠ˥p)(ElasticArray(zero(Gᶠ˥p.data))),
    typeof(Gᵇ˥p)(ElasticArray(zero(Gᵇ˥p.data))),
    typeof(Gᵇ˥p)(ElasticArray(zero(Gᵇ˥p.data))),
    freqs,fermi,bose,boltz)
  end
end

function f_vert_time(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ₀,σ,hᵧ,γ,δ₀ = model
  
  # real-time collision integral
  ∫dt(i,j,A,B) = sign(j-i) * integrate(times[min(i,j):max(i,j)], 
    [A[t1,t] * B[t,t2] for t=min(i,j):max(i,j)], TrapezoidalFast())
  
  calculate_Σ_time!(model, data, t1, t2)
  
  # conduction electron
  dGᶜL = -1.0im * (-1.0im * σ * GᶜL[t1,t2])
  dGᶜG = -1.0im * (-1.0im * σ * GᶜG[t1,t2])
  
  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f-λ₀) * GᶠL[t1,t2] + ∫dt(1,t1,ΣᶠG,GᶠL) - ∫dt(1,t2,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f-λ₀) * GᶠG[t1,t2] + ∫dt(t2,t1,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (-λ₀ * GᵇL[t1,t2] + ∫dt(1,t1,ΣᵇG,GᵇL) - ∫dt(1,t2,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (-λ₀ * GᵇG[t1,t2] + ∫dt(t2,t1,ΣᵇG,GᵇG))
  
  # thz pulse
  dGᵞL = -1.0im * ((hᵧ(times[t1])
   + 1.0im * γ(times[t1]) - 1.0im * δ₀) * GᵞL[t1,t2])
  dGᵞG = -1.0im * ((hᵧ(times[t1])
   + 1.0im * γ(times[t1]) - 1.0im * δ₀) * GᵞG[t1,t2])
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_time(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack hᵧ,γ,λ,δ₀ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t1,t] * B[t,t1] for t=1:t1], TrapezoidalFast())

  # commutator
  ⋊(a,b) = a * b - b * adjoint(a)

  calculate_Σ_time!(model, data, t1, t1)

  # conduction electron
  dGᶜL = 0.0
  dGᶜG = 0.0

  # pseudo fermion
  dGᶠL = -1.0im * (∫dt(ΣᶠG,GᶠL) - ∫dt(ΣᶠL,GᶠG) + ∫dt(GᶠL,ΣᶠG) - ∫dt(GᶠG,ΣᶠL))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (∫dt(ΣᵇG,GᵇL) - ∫dt(ΣᵇL,GᵇG) + ∫dt(GᵇL,ΣᵇG) - ∫dt(GᵇG,ΣᵇL))
  dGᵇG = 0.0
  
  # thz pulse
  dGᵞL = -1.0im * (hᵧ(times[t1]) ⋊ GᵞL[t1,t1] 
    + 1.0im * γ(times[t1]) * (GᵞL[t1,t1] + GᵞG[t1,t1]))
  dGᵞG = -1.0im * (hᵧ(times[t1]) ⋊ GᵞG[t1,t1] 
    + 1.0im * λ(times[t1]) * (GᵞG[t1,t1] - GᵞL[t1,t1]) 
    + 2.0im * γ(times[t1]) * GᵞG[t1,t1])
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_vert_freq(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,Σᶠ˥p,Σᶠ˥m,Σᵇ˥p,Σᵇ˥m,freqs,fermi,bose,boltz = data

  # real-frequency collision integral
  ∫dω(f) = integrate(freqs.mesh, f, TrapezoidalEvenFast()) / (2π)

  # conduction electron
  dGᶜL = 0.0
  dGᶜG = 0.0
  
  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(fermi .* (Σᶠ˥m[t1] .* conj(Gᶠ˥p[t2]) - Σᶠ˥p[t1] .* conj(Gᶠ˥m[t2]))))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    -∫dω(bose .* (Σᵇ˥m[t1] .* conj(Gᵇ˥p[t2]) - Σᵇ˥p[t1] .* conj(Gᵇ˥m[t2]))))
  dGᵇG = 0.0
  
  # thz pulse
  dGᵞL = 0.0
  dGᵞG = 0.0
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,Σᶠ˥p,Σᶠ˥m,Σᵇ˥p,Σᵇ˥m,freqs,fermi,bose,boltz = data

  # real-frequency collision integral
  ∫dω(f) = integrate(freqs.mesh, f, TrapezoidalEvenFast()) / (2π)

  # conduction electron: thermal contributions ∝ ζ → 0
  dGᶜL = 0.0
  dGᶜG = 0.0

  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(fermi .* (Σᶠ˥m[t1] .* conj(Gᶠ˥p[t1]) .- Σᶠ˥p[t1] .* conj(Gᶠ˥m[t1])))
    -∫dω(fermi .* (Gᶠ˥m[t1] .* conj(Σᶠ˥p[t1]) .- Gᶠ˥p[t1] .* conj(Σᶠ˥p[t1])))
    )
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    -∫dω(bose .* (Σᵇ˥m[t1] .* conj(Gᵇ˥p[t1]) .- Σᵇ˥p[t1] .* conj(Gᵇ˥m[t1])))
    +∫dω(bose .* (Gᵇ˥m[t1] .* conj(Σᵇ˥p[t1]) .- Gᵇ˥p[t1] .* conj(Σᵇ˥m[t1])))
    )
  dGᵇG = 0.0
  
  # thz pulse: no thermal contributions
  dGᵞL = 0.0
  dGᵞG = 0.0
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_line_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack Gᶜ˥m,Gᶜ˥p,Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,ΣᶠG,ΣᵇG,Σᶠ˥p,Σᶠ˥m,Σᵇ˥p,Σᵇ˥m = data
  @unpack ε₀f,λ₀,σ = model

  calculate_Σ_time!(model, data, t1, 1)
  calculate_Σ_freq!(model, data, t1)
  
  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t1,t] .* B[t] for t=1:t1], TrapezoidalFast())
  
  # conduction electron
  dGᶜ˥m = -1.0im * (-1.0im .* σ .* Gᶜ˥m[t1])
  dGᶜ˥p = -1.0im * (-1.0im .* σ .* Gᶜ˥p[t1])

  # pseudo fermion
  dGᶠ˥m = -1.0im * ((ε₀f-λ₀).*Gᶠ˥m[t1] .+ ∫dt(ΣᶠG, Gᶠ˥m) .+ Σᶠ˥m[t1].*Gᶠ˥m[1])
  dGᶠ˥p = -1.0im * ((ε₀f-λ₀).*Gᶠ˥p[t1] .+ ∫dt(ΣᶠG, Gᶠ˥p) .+ Σᶠ˥p[t1].*Gᶠ˥p[1])

  # slave boson
  dGᵇ˥m = -1.0im * (-λ₀.*Gᵇ˥m[t1] .+ ∫dt(ΣᵇG, Gᵇ˥m) .+ Σᵇ˥m[t1].*Gᵇ˥m[1])
  dGᵇ˥p = -1.0im * (-λ₀.*Gᵇ˥p[t1] .+ ∫dt(ΣᵇG, Gᵇ˥p) .+ Σᵇ˥p[t1].*Gᵇ˥p[1])

  return [dGᶜ˥m, dGᶜ˥p, dGᶠ˥m, dGᶠ˥p, dGᵇ˥m, dGᵇ˥p]
end

function calculate_Σ_time!(model::PhotonAssitedModel,data::ProblemData,i,j)
  @unpack V₀,Γ₀ = model
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  
  if size(ΣᶠL,1) < max(i,j)
    n = size(GᶜL,1)
    resize!(ΣᶠL,n)
    resize!(ΣᶠG,n)
    resize!(ΣᵇL,n)
    resize!(ΣᵇG,n)
  end
  
  ΣᶠL[i,j] = 1 * GᶜL[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᵇL[i,j]
  ΣᶠG[i,j] = 1 * GᶜG[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᵇG[i,j]
  ΣᵇL[i,j] = 2 * GᶠL[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᶜG[j,i]
  ΣᵇG[i,j] = 2 * GᶠG[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᶜL[j,i]
end

function calculate_Σ_freq!(model::PhotonAssitedModel,data::ProblemData,i)
  @unpack V₀ = model
  @unpack Gᶜ˥m,Gᶜ˥p,Gᶠ˥m,Gᶠ˥p,Gᵇ˥m,Gᵇ˥p,Σᶠ˥p,Σᶠ˥m,Σᵇ˥p,Σᵇ˥m,freqs,fermi = data
    
  if size(Σᶠ˥p,2) < i
    n = size(Gᶜ˥m,2)
    resize!(Σᶠ˥p,n)
    resize!(Σᶠ˥m,n)
    resize!(Σᵇ˥p,n)
    resize!(Σᵇ˥m,n)
  end
  
  Aᶜp = fermi .* conj(Gᶜ˥p[i] .- Gᶜ˥m[i])
  Aᶜh = (1 .- fermi) .* (Gᶜ˥m[i] .- Gᶜ˥p[i])
  
  Σᶠ˥m[i] = 1 * V₀^2 .* NCA.conv(freqs, Aᶜh, Gᵇ˥m[i]) / (2π)
  Σᶠ˥p[i] = 1 * V₀^2 .* NCA.conv(freqs, Aᶜh, Gᵇ˥p[i]) / (2π)
  Σᵇ˥m[i] = 2 * V₀^2 .* NCA.xcorr(freqs, Aᶜp, Gᶠ˥m[i]) / (2π)
  Σᵇ˥p[i] = 2 * V₀^2 .* NCA.xcorr(freqs, Aᶜp, Gᶠ˥p[i]) / (2π)
end

function NumericalIntegration._zero(x, y::Vector{<:AbstractVector})
  ret = zero(first(y)) # NOTE: not type-stable
end
