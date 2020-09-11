using NCA
using KadanoffBaym
using NumericalIntegration
using ElasticArrays
using Parameters
using Test

@with_kw mutable struct PhotonAssitedModel
  # single impurity Anderson model parameters
  V₀::Float64     # hybridization strength
  ε₀f::Float64    # f-electron ground state energy
  λ₀::Float64=NaN # gauge chemical potential
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
  
  Aᶜ˹::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  GᶠT::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  GᶠM::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  GᵇT::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  GᵇM::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  
  ΣᶠL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  ΣᶠG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  ΣᵇL::GreenFunction{ComplexF64, Matrix{ComplexF64}, Lesser}
  ΣᵇG::GreenFunction{ComplexF64, Matrix{ComplexF64}, Greater}
  
  ΣᶠT::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  ΣᶠM::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  ΣᵇT::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  ΣᵇM::GreenFunction{ComplexF64, ElasticArray{ComplexF64,2,1,Vector{ComplexF64}}, MixedGreater}
  
  freqs::NCA.AbstractMesh
  fermi::Vector{Float64}
  fermi⁻::Vector{Float64}
  th½::Vector{Float64}
end

function ProblemData(model::PhotonAssitedModel, β::Real, mesh, nᵞ::Real)
  # Solve NCA in Equilibrium
  nca = FreqNCA(
    model=NCA.SIAM(
      V₀=model.V₀,
      ε₀f=model.ε₀f,
      tilde=true,
      Nc₀=x->2pi * NCA.lorentzian(x, 0.0, model.σ)),
    β=β,
    mesh=mesh)
  
  imΣ_t = NCA.solve!(nca; project=false)
  model.λ₀ = nca.model.λ₀

  Ginv = NCA.slave_particle_inv_gf(nca, imΣ_t)
  GᵇA = 1.0 ./ Ginv[1]
  GᶠA = 1.0 ./ Ginv[2]

  # Trim just the imaginary part
  imGᵇ = view(reinterpret(Float64, GᵇA), 2:2:length(GᵇA))
  imGᵇ[imGᵇ .< nca.eps] .= 0.0
  imGᶠ = view(reinterpret(Float64, GᶠA), 2:2:length(GᶠA))
  imGᶠ[imGᶠ .< nca.eps] .= 0.0

  # Create a new frequency grid that supports for complex convolutions
  freqs = NCA.LinearMesh(nca.mesh.xmin, nca.mesh.xmax, nca.mesh.L, true);

  # Integral over frequencies
  ∫dω(f) = integrate(freqs.mesh, f, TrapezoidalEvenFast()) / (2π)

  @unpack fermi,fermi⁻,th½ = nca.data

  Aᶜ = 1im * nca.data.Nc
  GᶜL = GreenFunction(∫dω(fermi .* Aᶜ) * ones(1,1), Lesser)
  GᶜG = GreenFunction(-∫dω(fermi⁻ .* Aᶜ) * ones(1,1), Greater)
  Aᶜ˹ = GreenFunction(ElasticArray(hcat(Aᶜ)), MixedGreater)

  Aᶠ_t = 2im * nca.data.imG_t[:,2]
  GᶠL = GreenFunction(∫dω(fermi .* Aᶠ_t) * ones(1,1), Lesser)
  GᶠG = GreenFunction(-(1.0im) * ones(1,1), Greater)
  GᶠT = GreenFunction(ElasticArray(hcat(Aᶠ_t)), MixedGreater)
  GᶠM = GreenFunction(ElasticArray(hcat(GᶠA)), MixedGreater)

  Aᵇ_t = 2im * nca.data.imG_t[:,1]
  GᵇL = GreenFunction(-∫dω(fermi .* Aᵇ_t) * ones(1,1), Lesser)  
  GᵇG = GreenFunction(-(1.0im) * ones(1,1), Greater)
  GᵇT = GreenFunction(ElasticArray(hcat(Aᵇ_t)), MixedGreater)
  GᵇM = GreenFunction(ElasticArray(hcat(GᵇA)), MixedGreater)

  GᵞL = GreenFunction(-1.0im * nᵞ * ones(1,1), Lesser)  
  GᵞG = GreenFunction(-1.0im * (1 + nᵞ) * ones(1,1), Greater)

  # Spectral report
  println("\nEquilibrium spectral function normalizations:")
  println(" ∫dω Aᶜ(ω) = $(imag(∫dω(Aᶜ))), nᶜ = $(imag(GᶜL[1,1]))")
  println(" ∫dω Aᶠ(ω) = $(imag(∫dω(fermi⁻ .* Aᶠ_t))), nᶠ = $(imag(GᶠL[1,1]))")
  println(" ∫dω Aᵇ(ω) = $(imag(∫dω(fermi⁻ .* Aᵇ_t))), nᵇ = $(imag(-GᵇL[1,1]))")
  println(" Q = $(imag(2GᶠL[1,1] - GᵇL[1,1]))")

  data = ProblemData(GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    Aᶜ˹,GᶠT,GᶠM,GᵇT,GᵇM,
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    typeof(GᶠT)(ElasticArray(zero(GᶠT.data))),
    typeof(GᶠM)(ElasticArray(zero(GᶠM.data))),
    typeof(GᵇT)(ElasticArray(zero(GᵇT.data))),
    typeof(GᵇM)(ElasticArray(zero(GᵇM.data))),
    freqs,fermi,fermi⁻,th½)

  # NOTE: initialize self-energies
  calculate_Σ_time!(model, data, 1, 1)
  calculate_Σ_freq!(model, data, 1)

  @testset "Σ equivalence between mixed and equilibrium formulations" begin
    @test data.ΣᵇT[1] ≈ +2im * imΣ_t[:,1]
    @test data.ΣᵇM[1] |> imag ≈ imΣ_t[:,1] .* fermi⁻
    @test data.ΣᵇM[1] |> real ≈ NCA.kramerskronig(nca.mesh, imΣ_t[:,1] .* fermi⁻, :real) rtol=5e-2
    @test data.ΣᶠT[1] ≈ +2im * imΣ_t[:,2]
    @test data.ΣᶠM[1] |> imag ≈ imΣ_t[:,2] .* fermi⁻
    @test data.ΣᶠM[1] |> real ≈ NCA.kramerskronig(nca.mesh, imΣ_t[:,2] .* fermi⁻, :real) rtol=5e-2
  end

  return data
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
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,freqs,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(freqs.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)

  # conduction electron
  dGᶜL = 0.0
  dGᶜG = 0.0
  
  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(conj(ΣᶠM[t1]) .* GᶠT[t2] - conj(ΣᶠT[t1]) .* GᶠM[t2]))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    -∫dω(conj(ΣᵇM[t1]) .* GᵇT[t2] - conj(ΣᵇT[t1]) .* GᵇM[t2]))
  dGᵇG = 0.0
  
  # thz pulse
  dGᵞL = 0.0
  dGᵞG = 0.0
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,freqs,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(freqs.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)

  # conduction electron: thermal contributions ∝ ζ → 0
  dGᶜL = 0.0
  dGᶜG = 0.0

  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(conj(ΣᶠM[t1]) .* GᶠT[t1] - conj(ΣᶠT[t1]) .* GᶠM[t1])
    -∫dω(conj(GᶠM[t1]) .* ΣᶠT[t1] - conj(GᶠT[t1]) .* ΣᶠM[t1]))

  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    -∫dω(conj(ΣᵇM[t1]) .* GᵇT[t1] - conj(ΣᵇT[t1]) .* GᵇM[t1])
    +∫dω(conj(GᵇM[t1]) .* ΣᵇT[t1] - conj(GᵇT[t1]) .* ΣᵇM[t1]))
  dGᵇG = 0.0
  
  # thz pulse: no thermal contributions
  dGᵞL = 0.0
  dGᵞG = 0.0

  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_line_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack Aᶜ˹,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG = data
  @unpack ε₀f,λ₀,σ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t] .* B[t,t1] for t=1:t1], TrapezoidalFast())
  
  # conduction electron
  dAᶜ˹ = +1.0im * (+1.0im .* σ .* Aᶜ˹[t1])

  # pseudo fermion
  dGᶠT = +1.0im * ((ε₀f-λ₀).*GᶠT[t1] .- ∫dt(GᶠT, ΣᶠG) .+ GᶠT[1] .* ΣᶠM[t1] .+ conj(GᶠM[1]) .* ΣᶠT[t1])
  dGᶠM = +1.0im * ((ε₀f-λ₀).*GᶠM[t1] .- ∫dt(GᶠM, ΣᶠG) .+ GᶠM[1] .* ΣᶠM[t1])

  # slave boson
  dGᵇT = +1.0im * (-λ₀.*GᵇT[t1] .- ∫dt(GᵇT, ΣᵇG) .+ GᵇT[1] .* ΣᵇM[t1] .+ conj(GᵇM[1]) .* ΣᵇT[t1])
  dGᵇM = +1.0im * (-λ₀.*GᵇM[t1] .- ∫dt(GᵇM, ΣᵇG) .+ GᵇM[1] .* ΣᵇM[t1])

  return [dAᶜ˹, dGᶠT, dGᶠM, dGᵇT, dGᵇM]
end

function calculate_Σ_time!(model::PhotonAssitedModel,data::ProblemData,t1,t2)
  @unpack V₀,Γ₀ = model
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  
  if (n = size(GᶜL,1)) > size(ΣᶠL,1)
    resize!(ΣᶠL,n)
    resize!(ΣᶠG,n)
    resize!(ΣᵇL,n)
    resize!(ΣᵇG,n)
  end
  
  ΣᶠL[t1,t2] = +1.0im * GᶜL[t1,t2] * (V₀^2 + Γ₀^2 * (GᵞL[t1,t2] + GᵞG[t2,t1])) * GᵇL[t1,t2]
  ΣᶠG[t1,t2] = +1.0im * GᶜG[t1,t2] * (V₀^2 + Γ₀^2 * (GᵞL[t1,t2] + GᵞG[t2,t1])) * GᵇG[t1,t2]
  ΣᵇL[t1,t2] = -2.0im * GᶠL[t1,t2] * (V₀^2 + Γ₀^2 * (GᵞL[t1,t2] + GᵞG[t2,t1])) * GᶜG[t2,t1]
  ΣᵇG[t1,t2] = -2.0im * GᶠG[t1,t2] * (V₀^2 + Γ₀^2 * (GᵞL[t1,t2] + GᵞG[t2,t1])) * GᶜL[t2,t1]
end

function calculate_Σ_freq!(model::PhotonAssitedModel,data::ProblemData,t1)
  @unpack V₀ = model
  @unpack Aᶜ˹,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack freqs, fermi, fermi⁻, th½ = data
    
  if (n = size(Aᶜ˹,2)) > size(ΣᶠT,2)
    resize!(ΣᶠT,n)
    resize!(ΣᶠM,n)
    resize!(ΣᵇT,n)
    resize!(ΣᵇM,n)
  end
  
  ΣᶠT[t1] = -0.5im * V₀^2 / (2π) .* (
    NCA.conv(freqs, Aᶜ˹[t1], GᵇT[t1]) + NCA.conv(freqs, th½ .* Aᶜ˹[t1], th½ .* GᵇT[t1]))
  ΣᶠM[t1] = -1.0im * V₀^2 / (2π) .* NCA.conv(freqs, fermi⁻ .* Aᶜ˹[t1], GᵇM[t1])
  
  ΣᵇT[t1] = +1.0im * V₀^2 / (2π) .* (
    NCA.xcorr(freqs, Aᶜ˹[t1], GᶠT[t1]) - NCA.xcorr(freqs, th½ .* Aᶜ˹[t1], th½ .* GᶠT[t1]))
  ΣᵇM[t1] = +2.0im * V₀^2 / (2π) .* NCA.xcorr(freqs, fermi .* Aᶜ˹[t1], GᶠM[t1])
  
end

function NumericalIntegration._zero(x, y::Vector{<:AbstractVector})
  ret = zero(first(y)) # NOTE: not type-stable
end
