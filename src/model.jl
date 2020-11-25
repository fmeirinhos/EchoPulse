using NCA, NLsolve
using KadanoffBaym, ElasticArrays
using NumericalIntegration
using Parameters: @unpack
using Test

Base.@kwdef mutable struct PhotonAssitedModel
  # single impurity Anderson model parameters
  V₀::Float64     # hybridization strength
  ε₀f::Float64    # f-electron ground state energy
  λ₀::Float64=NaN # gauge chemical potential
  σ::Float64      # c-electron lorenztian bandwidth

  # pulse parameters
  Γ₀::Float64     # light-matter coupling strength
  ω₀::Float64     # photon ground state energy
  δ₀::Float64     # photon energy bandwidth
  λ=nothing       # loss function
  γ=nothing       # pump function
end

struct ProblemData{T,U}
  GᶜL::GreenFunction{T, Array{T,2}, Lesser}
  GᶜG::GreenFunction{T, Array{T,2}, Greater}
  GᶠL::GreenFunction{T, Array{T,2}, Lesser}
  GᶠG::GreenFunction{T, Array{T,2}, Greater}
  GᵇL::GreenFunction{T, Array{T,2}, Lesser}
  GᵇG::GreenFunction{T, Array{T,2}, Greater}
  GᵞL::GreenFunction{T, Array{T,2}, Lesser}
  GᵞG::GreenFunction{T, Array{T,2}, Greater}
  
  ΣᶠL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᶠG::GreenFunction{T, Array{T,2}, Greater}
  ΣᵇL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᵇG::GreenFunction{T, Array{T,2}, Greater}

  GᶜA::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  GᶠT::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  GᶠM::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  GᵇT::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  GᵇM::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  
  ΣᶠT::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  ΣᶠM::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  ΣᵇT::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  ΣᵇM::GreenFunction{T, <:ElasticArray{T,2}, MixedGreater}
  
  mesh::LinearMesh{U}
  fermi::Vector{U}
  fermi⁻::Vector{U}
  th½::Vector{U}
end

function ProblemData(model::PhotonAssitedModel, β::Real, mesh, nᵞ::Real)
  # Get the correlated tilde and non-tilde Green functions
  imG_t, G = begin

    # Equilibrium model
    nca = FreqNCA(
      model=SIAM(
        V₀=model.V₀,
        ε₀f=model.ε₀f,
        tilde=true,
        Nc₀=x->2pi * NCA.lorentzian(x, -0.0, model.σ)),
      β=β,
      mesh=mesh)

    # Find the NCA fixed point
    imΣ_t = nlsolve(nca.data.imΣ_t, method=:anderson, beta=0.9, show_trace=:true) do imΣ
      imG = NCA.slave_particle_imG(nca, imΣ; calculate_λ=true)
      imΣ′ = NCA.slave_particle_imΣ(nca, imG)
      imΣ′ - imΣ
    end.zero
    
    # Get observables
    imG_t = NCA.slave_particle_imG(nca, imΣ_t)
    G = 1 ./ NCA.slave_particle_inv_gf(nca, NCA.slave_particle_Σ(nca, imΣ_t))
    imG_t, G
  end

  # Update gauge chem potential
  model.λ₀ = nca.model.λ

  # Create a new frequency grid that supports complex convolutions
  mesh = NCA.LinearMesh(nca.mesh.xmin, nca.mesh.xmax, nca.mesh.L, true);
  ∫dω(f) = integrate(mesh.mesh, f, TrapezoidalEvenFast()) / (2π)

  @unpack fermi,fermi⁻,th½, Nc = nca.data

  GᶜL = GreenFunction(ones(1,1) * (+1.0im * ∫dω(fermi .* Nc)), Lesser)
  GᶜG = GreenFunction(ones(1,1) * (-1.0im * ∫dω(fermi⁻ .* Nc)), Greater)
  GᶜA = GreenFunction(ElasticArray(hcat(1.0im * Nc)), MixedGreater)

  GᶠL = GreenFunction(ones(1,1) * (+1.0im * ∫dω(fermi .* 2imG_t[:,2])), Lesser)
  GᶠG = GreenFunction(ones(1,1) * (-1.0im), Greater)
  GᶠT = GreenFunction(ElasticArray(hcat(2im * imG_t[:,2])), MixedGreater)
  GᶠM = GreenFunction(ElasticArray(hcat(G[:,2])), MixedGreater)

  GᵇL = GreenFunction(ones(1,1) * (-1.0im * ∫dω(fermi .* 2imG_t[:,1])), Lesser)  
  GᵇG = GreenFunction(ones(1,1) * (-1.0im), Greater)
  GᵇT = GreenFunction(ElasticArray(hcat(2im * imG_t[:,1])), MixedGreater)
  GᵇM = GreenFunction(ElasticArray(hcat(G[:,1])), MixedGreater)

  GᵞL = GreenFunction(ones(1,1) * (-1.0im * nᵞ), Lesser)
  GᵞG = GreenFunction(ones(1,1) * (-1.0im * (1 + nᵞ)), Greater)

  # Spectral report
  println("\nEquilibrium spectral function normalizations:")
  println(" ∫dω f(ω) Aᶜ(ω) = $(imag(+GᶜL[1,1]))")
  println(" ∫dω f(ω) Aᶠ(ω) = $(imag(+GᶠL[1,1]))")
  println(" ∫dω b(ω) Aᵇ(ω) = $(imag(-GᵇL[1,1]))")
  println(" Q = $(imag(2GᶠL[1,1] - GᵇL[1,1]))")

  data = ProblemData{ComplexF64,Float64}(
    GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,
    typeof(GᶠT)(ElasticArray(zero(GᶠT.data))),
    typeof(GᶠM)(ElasticArray(zero(GᶠM.data))),
    typeof(GᵇT)(ElasticArray(zero(GᵇT.data))),
    typeof(GᵇM)(ElasticArray(zero(GᵇM.data))),
    mesh,fermi,fermi⁻,th½)

  @testset "Σ equivalence between mixed and equilibrium formulations" begin
    calculate_Σ_time!(model, data, 1, 1)
    calculate_Σ_freq!(model, data, 1)
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
  @unpack ε₀f,λ₀,σ,ω₀,δ₀,λ,γ = model

  # real-time collision integral. NOTE: @assert t1 >= t2
  ∫dt(i,j,A,B) = sign(j-i) * integrate(times[min(i,j):max(i,j)], 
    [A[t1,t] * B[t,t2] for t=min(i,j):max(i,j)], TrapezoidalFast())

  # conduction electron
  dGᶜL = -1.0im * (-0.0 * GᶜL[t1,t2] - 1.0im * σ * GᶜL[t1,t2])
  dGᶜG = -1.0im * (-0.0 * GᶜG[t1,t2] - 1.0im * σ * GᶜG[t1,t2])
  
  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f + λ₀) * GᶠL[t1,t2] + ∫dt(1,t1,ΣᶠG,GᶠL) - ∫dt(1,t2,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f + λ₀) * GᶠG[t1,t2] + ∫dt(t2,t1,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (λ₀ * GᵇL[t1,t2] + ∫dt(1,t1,ΣᵇG,GᵇL) - ∫dt(1,t2,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (λ₀ * GᵇG[t1,t2] + ∫dt(t2,t1,ΣᵇG,GᵇG))

  # thz pulse
  dGᵞL = -0.0im * (ω₀ + 0.5im * (γ(times[t1]) - λ(times[t1])
   - 1.0im * δ₀) * GᵞL[t1,t2])::ComplexF64
  dGᵞG = -0.0im * (ω₀ + 0.5im * (γ(times[t1]) - λ(times[t1])
   - 1.0im * δ₀) * GᵞG[t1,t2])::ComplexF64
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_diag_time(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ω₀,λ,γ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times[1:t1], [A[t1,t] * B[t,t1] for t=1:t1], TrapezoidalFast())

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
  dGᵞL = -0.0im * ((ω₀ - 0.5im * (γ(times[t1]) + λ(times[t1]))) ⋊ GᵞL[t1,t1] 
    + 1.0im * γ(times[t1]) * (GᵞL[t1,t1] + GᵞG[t1,t1]))::ComplexF64
  dGᵞG = -0.0im * ((ω₀ - 0.5im * (γ(times[t1]) + λ(times[t1]))) ⋊ GᵞG[t1,t1] 
    + 1.0im * λ(times[t1]) * (GᵞG[t1,t1] - GᵞL[t1,t1]) 
    + 2.0im * γ(times[t1]) * GᵞG[t1,t1])::ComplexF64
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG]
end

function f_vert_freq(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(mesh.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)

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
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(mesh.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)

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
  @unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG = data
  @unpack ε₀f,λ₀,σ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [-A[t] .* B[t1,t]' for t=1:t1], TrapezoidalFast())
  
  # conduction electron
  dGᶜA = +1.0im * (+1.0im .* σ .* GᶜA[t1])

  # pseudo fermion
  dGᶠT = +1.0im * ((ε₀f + λ₀).*GᶠT[t1] .- ∫dt(GᶠT, ΣᶠG) .+ GᶠT[1] .* ΣᶠM[t1] .+ conj(GᶠM[1]) .* ΣᶠT[t1])
  dGᶠM = +1.0im * ((ε₀f + λ₀).*GᶠM[t1] .- ∫dt(GᶠM, ΣᶠG) .+ GᶠM[1] .* ΣᶠM[t1])

  # slave boson
  dGᵇT = +1.0im * (λ₀ .* GᵇT[t1] .- ∫dt(GᵇT, ΣᵇG) .+ GᵇT[1] .* ΣᵇM[t1] .+ conj(GᵇM[1]) .* ΣᵇT[t1])
  dGᵇM = +1.0im * (λ₀ .* GᵇM[t1] .- ∫dt(GᵇM, ΣᵇG) .+ GᵇM[1] .* ΣᵇM[t1])

  return [dGᶜA, dGᶠT, dGᶠM, dGᵇT, dGᵇM]
end

function calculate_Σ_time!(model::PhotonAssitedModel,data::ProblemData,t1,t2)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack V₀,Γ₀ = model
  
  if (n = size(GᶜL,2)) > size(ΣᶠL,2)
    resize!(ΣᶠL,n)
    resize!(ΣᶠG,n)
    resize!(ΣᵇL,n)
    resize!(ΣᵇG,n)
  end

  V² = V₀^2 + 1.0im * Γ₀^2 * (GᵞL[t1,t2] + GᵞG[t1,t2])

  ΣᶠL[t1,t2] = +1.0im * V² * GᵇL[t1,t2] * GᶜL[t1,t2]
  ΣᶠG[t1,t2] = +1.0im * V² * GᵇG[t1,t2] * GᶜG[t1,t2]
  ΣᵇL[t1,t2] = -2.0im * V² * GᶠL[t1,t2] * GᶜG[t2,t1]
  ΣᵇG[t1,t2] = -2.0im * V² * GᶠG[t1,t2] * GᶜL[t2,t1]
end

function calculate_Σ_freq!(model::PhotonAssitedModel,data::ProblemData,t1)
  @unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack mesh, fermi, fermi⁻, th½ = data
  @unpack V₀ = model
  
  if (n = size(GᶜA,2)) > size(ΣᶠT,2)
    resize!(ΣᶠT,n)
    resize!(ΣᶠM,n)
    resize!(ΣᵇT,n)
    resize!(ΣᵇM,n)
  end
  
  ΣᶠT[t1] = -0.5im * V₀^2 / (2π) .* (
    NCA.conv(mesh, GᶜA[t1], GᵇT[t1]) + NCA.conv(mesh, th½ .* GᶜA[t1], th½ .* GᵇT[t1]))
  ΣᶠM[t1] = -1.0im * V₀^2 / (2π) .* NCA.conv(mesh, fermi⁻ .* GᶜA[t1], GᵇM[t1])
  
  ΣᵇT[t1] = +1.0im * V₀^2 / (2π) .* (
    NCA.xcorr(mesh, GᶜA[t1], GᶠT[t1]) - NCA.xcorr(mesh, th½ .* GᶜA[t1], th½ .* GᶠT[t1]))
  ΣᵇM[t1] = +2.0im * V₀^2 / (2π) .* NCA.xcorr(mesh, fermi .* GᶜA[t1], GᶠM[t1])
end

function NumericalIntegration._zero(x, y::Vector{<:AbstractVector})
  ret = zero(first(y)) # NOTE: not type-stable
end
