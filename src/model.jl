using NCA
using KadanoffBaym
using NumericalIntegration
using ElasticArrays
using Parameters

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
  GᶜL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  GᶜG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  GᶠL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  GᶠG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  GᵇL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  GᵇG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  GᵞL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  GᵞG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  
  Aᶜ˹::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Aᶠ˹::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Gᶠ˹p::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Aᵇ˹::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Gᵇ˹p::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  
  ΣᶠL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  ΣᶠG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  ΣᵇL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  ΣᵇG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  
  Σᶠ˹A::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Σᶠ˹p::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Σᵇ˹A::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  Σᵇ˹p::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedLesser}
  
  freqs::NCA.AbstractMesh
  fermi::Vector{Float64}
  fermi⁻::Vector{Float64}
  th½::Vector{Float64}

  function ProblemData(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    Aᶜ˹,Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,freqs,fermi,fermi⁻,th½)

  new(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    Aᶜ˹,Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    typeof(Aᶠ˹)(ElasticArray(zero(Aᶠ˹.data))),
    typeof(Gᶠ˹p)(ElasticArray(zero(Gᶠ˹p.data))),
    typeof(Aᵇ˹)(ElasticArray(zero(Aᵇ˹.data))),
    typeof(Gᵇ˹p)(ElasticArray(zero(Gᵇ˹p.data))),
    freqs,fermi,fermi⁻,th½)
  end
end

function f_vert_time(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ₀,σ,hᵧ,γ,δ₀ = model
  
  # real-time collision integral
  ∫dt(i,j,A,B) = sign(j-i) * integrate(times[min(i,j):max(i,j)], 
    [A[t1,t] * B[t,t2] for t=min(i,j):max(i,j)], TrapezoidalFast())

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
   + 1.0im * γ(times[t1]) - 1.0im * δ₀) * GᵞL[t1,t2])::ComplexF64
  dGᵞG = -1.0im * ((hᵧ(times[t1])
   + 1.0im * γ(times[t1]) - 1.0im * δ₀) * GᵞG[t1,t2])::ComplexF64
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_time(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack hᵧ,γ,λ,δ₀ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t1,t] * B[t,t1] for t=1:t1], TrapezoidalFast())

  # commutator
  ⋊(a,b) = a * b - b * adjoint(a)

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
    + 1.0im * γ(times[t1]) * (GᵞL[t1,t1] + GᵞG[t1,t1]))::ComplexF64
  dGᵞG = -1.0im * (hᵧ(times[t1]) ⋊ GᵞG[t1,t1] 
    + 1.0im * λ(times[t1]) * (GᵞG[t1,t1] - GᵞL[t1,t1]) 
    + 2.0im * γ(times[t1]) * GᵞG[t1,t1])::ComplexF64
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_vert_freq(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,Σᶠ˹A,Σᶠ˹p,Σᵇ˹A,Σᵇ˹p,freqs,fermi = data

  # real-frequency collision integral
  ∫dω(f) = (f[abs.(f) .< 1e-13] .= 0.0; 
    integrate(freqs.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π))

  # conduction electron
  dGᶜL = 0.0
  dGᶜG = 0.0
  
  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(conj(Σᶠ˹p[t1]) .* Aᶠ˹[t2] - conj(Σᶠ˹A[t1]) .* Gᶠ˹p[t2]))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    +∫dω(conj(Σᵇ˹p[t1]) .* Aᵇ˹[t2] - conj(Σᵇ˹A[t1]) .* Gᵇ˹p[t2]))
  dGᵇG = 0.0
  
  # thz pulse
  dGᵞL = 0.0
  dGᵞG = 0.0
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,Σᶠ˹A,Σᶠ˹p,Σᵇ˹A,Σᵇ˹p,freqs,fermi = data

  # real-frequency collision integral
  ∫dω(f) = (f[abs.(f) .< 1e-13] .= 0.0; 
    integrate(freqs.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π))

  # conduction electron: thermal contributions ∝ ζ → 0
  dGᶜL = 0.0
  dGᶜG = 0.0

  # pseudo fermion # TODO: review this formula
  dGᶠL = -1.0im * (
    +∫dω(conj(Σᶠ˹p[t1]) .* Aᶠ˹[t1] - conj(Σᶠ˹A[t1]) .* Gᶠ˹p[t1])
    +∫dω(conj(Gᶠ˹p[t1]) .* Σᶠ˹A[t1] - conj(Aᶠ˹[t1]) .* Σᶠ˹p[t1]))
    # +0.5∫dω(conj(Σᵇ˹p[t1]) .* Aᵇ˹[t1] - conj(Σᵇ˹A[t1]) .* Gᵇ˹p[t1]))

  dGᶠG = 0.0
  
  # slave boson # TODO: review this formula
  dGᵇL = -1.0im * (
    +∫dω(conj(Σᵇ˹p[t1]) .* Aᵇ˹[t1] - conj(Σᵇ˹A[t1]) .* Gᵇ˹p[t1])
    +∫dω(conj(Gᵇ˹p[t1]) .* Σᵇ˹A[t1] - conj(Aᵇ˹[t1]) .* Σᵇ˹p[t1]))
    # +2∫dω(conj(Σᶠ˹p[t1]) .* Aᶠ˹[t1] - conj(Σᶠ˹A[t1]) .* Gᶠ˹p[t1]))
  dGᵇG = 0.0
  
  # thz pulse: no thermal contributions
  dGᵞL = 0.0
  dGᵞG = 0.0

  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_line_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack Aᶜ˹,Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,Σᶠ˹A,Σᶠ˹p,Σᵇ˹A,Σᵇ˹p = data
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG = data
  @unpack ε₀f,λ₀,σ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t] .* B[t,t1] for t=1:t1], TrapezoidalFast())
  
  # conduction electron
  dAᶜ˹ = +1.0im * (+1.0im .* σ .* Aᶜ˹[t1])

  # pseudo fermion
  dAᶠ˹ = +1.0im * ((ε₀f-λ₀).*Aᶠ˹[t1] .- ∫dt(Aᶠ˹, ΣᶠG) .+ Σᶠ˹A[t1].*conj(Gᶠ˹p[1]) .+ Σᶠ˹p[t1].*Aᶠ˹[1])
  dGᶠ˹p = +1.0im * ((ε₀f-λ₀).*Gᶠ˹p[t1] .- ∫dt(Gᶠ˹p, ΣᶠG) .+ Σᶠ˹p[t1].*Gᶠ˹p[1])

  # slave boson
  dAᵇ˹ = +1.0im * (-λ₀.*Aᵇ˹[t1] .- ∫dt(Aᵇ˹, ΣᵇG) .+ Σᵇ˹A[t1].*conj(Gᵇ˹p[1]) .+ Σᵇ˹p[t1].*Aᵇ˹[1])
  dGᵇ˹p = +1.0im * (-λ₀.*Gᵇ˹p[t1] .- ∫dt(Gᵇ˹p, ΣᵇG) .+ Σᵇ˹p[t1].*Gᵇ˹p[1])

  return [dAᶜ˹, dAᶠ˹, dGᶠ˹p, dAᵇ˹, dGᵇ˹p]
end

function calculate_Σ_time!(model::PhotonAssitedModel,data::ProblemData,i,j)
  @unpack V₀,Γ₀ = model
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  
  if (n = size(GᶜL,1)) > size(ΣᶠL,1)
    resize!(ΣᶠL,n)
    resize!(ΣᶠG,n)
    resize!(ΣᵇL,n)
    resize!(ΣᵇG,n)
  end
  
  ΣᶠL[i,j] = +1.0im * GᶜL[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᵇL[i,j]
  ΣᶠG[i,j] = +1.0im * GᶜG[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᵇG[i,j]
  ΣᵇL[i,j] = -2.0im * GᶠL[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᶜG[j,i]
  ΣᵇG[i,j] = -2.0im * GᶠG[i,j] * (V₀^2 + Γ₀^2 * (GᵞL[i,j] + GᵞG[j,i])) * GᶜL[j,i]
end

function calculate_Σ_freq!(model::PhotonAssitedModel,data::ProblemData,i)
  @unpack V₀ = model
  @unpack Aᶜ˹,Aᶠ˹,Gᶠ˹p,Aᵇ˹,Gᵇ˹p,Σᶠ˹A,Σᶠ˹p,Σᵇ˹A,Σᵇ˹p = data
  @unpack freqs, fermi, fermi⁻, th½ = data
    
  if (n = size(Aᶜ˹,2)) > size(Σᶠ˹p,2)
    resize!(Σᶠ˹p,n)
    resize!(Σᶠ˹A,n)
    resize!(Σᵇ˹p,n)
    resize!(Σᵇ˹A,n)
  end
  
  Σᶠ˹p[i] = -1.0im * V₀^2 / (2π) .* NCA.conv(freqs, fermi⁻ .* Aᶜ˹[i], Gᵇ˹p[i])
  Σᶠ˹A[i] = -0.5im * V₀^2 / (2π) .* (
    NCA.conv(freqs, Aᶜ˹[i], Aᵇ˹[i]) + NCA.conv(freqs, th½ .* Aᶜ˹[i], th½ .* Aᵇ˹[i]))

  Σᵇ˹p[i] = +2.0im * V₀^2 / (2π) .* NCA.xcorr(freqs, fermi .* Aᶜ˹[i], Gᶠ˹p[i])
  Σᵇ˹A[i] = +1.0im * V₀^2 / (2π) .* (
    NCA.xcorr(freqs, Aᶜ˹[i], Aᶠ˹[i]) - NCA.xcorr(freqs, th½ .* Aᶜ˹[i], th½ .* Aᶠ˹[i]))
end

function NumericalIntegration._zero(x, y::Vector{<:AbstractVector})
  ret = zero(first(y)) # NOTE: not type-stable
end
