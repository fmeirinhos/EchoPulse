using NCA, NLsolve
using NCA.FFTW: fftfreq, fft
using KadanoffBaym
using NumericalIntegration
using Parameters: @unpack
using Test
using LinearAlgebra: diagind

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
  
  ΣᶠL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᶠG::GreenFunction{T, Array{T,2}, Greater}
  ΣᵇL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᵇG::GreenFunction{T, Array{T,2}, Greater}

  ts::Vector{Float64}
end

function ProblemData(model::PhotonAssitedModel, β::Real, mesh, nᵞ::Real)
  # Get the correlated tilde and non-tilde Green functions
  imG_t, G_ = begin

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
    Gc = complex.(NCA.kramerskronig(nca.mesh, 0.5 * nca.data.Nc, :real), 0.5 * nca.data.Nc) .* nca.data.fermi⁻
    hcat(imG_t, 0.5 * nca.data.Nc) ./ pi, hcat(G, Gc) ./ pi
  end

  # Update gauge chem potential
  model.λ₀ = nca.model.λ

  ωs = nca.mesh.mesh
  dω= ωs[2] - ωs[1]
  
  ts = fftfreq(length(ωs), 2π/dω)
  is = sortperm(ts)
  
  ℯⁱᵠ = dω * exp.(1.0im * ωs[1] .* ts)
  
  L = mapslices(imG_t, dims=1) do x
    im_ = x .* nca.data.fermi
    re_ = NCA.kramerskronig(nca.mesh, im_, :real)
    complex.(real(ℯⁱᵠ .* fft(re_)), real(ℯⁱᵠ .* fft(im_)))
  end
  G = mapslices(G_, dims=1) do x
    complex.(real(ℯⁱᵠ .* fft(real(x))), real(ℯⁱᵠ .* fft(imag(x))))
  end

  function G∞(g)
    Z_ = zeros(eltype(g), length(g), length(g))
    
    L2 = size(Z_,1) ÷ 2

    Z_[diagind(Z_, 0)] .= 1.0im * imag(g[1])

    g = g[is]
    for i in 1:(L2-1)
      view(Z_, diagind(Z_, +i)) .= g[L2+1+i]
      view(Z_, diagind(Z_, -i)) .= g[L2+1-i]
    end
    return Z_
  end

  GᶜL = GreenFunction(G∞(L[:,3]), Lesser)
  GᶜG = GreenFunction(-G∞(G[:,3]), Greater)

  GᶠL = GreenFunction(G∞(L[:,2]), Lesser)
  GᶠG = GreenFunction(-G∞(G[:,2]), Greater)

  GᵇL = GreenFunction(-G∞(L[:,1]), Lesser)  
  GᵇG = GreenFunction(-G∞(G[:,1]), Greater)

  # Spectral report
  println("\nEquilibrium spectral function normalizations:")
  println(" ∫dω f(ω) Aᶜ(ω) = $(imag(+GᶜL[1,1]))")
  println(" ∫dω f(ω) Aᶠ(ω) = $(imag(+GᶠL[1,1]))")
  println(" ∫dω b(ω) Aᵇ(ω) = $(imag(-GᵇL[1,1]))")
  println(" Q = $(imag(2GᶠL[1,1] - GᵇL[1,1]))")

  data = ProblemData{ComplexF64,Float64}(
    GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    ts[is] .- ts[is[end]])

  return data
end

function f_vert_time(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG = data #,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ₀,σ,ω₀,δ₀,λ,γ,V₀ = model

  ΣᶠL(t1,t2) = +1.0im * V₀^2 * GᵇL[t1,t2] * GᶜL[t1,t2]
  ΣᶠG(t1,t2) = +1.0im * V₀^2 * GᵇG[t1,t2] * GᶜG[t1,t2]
  ΣᵇL(t1,t2) = +2.0im * V₀^2 * GᶠL[t1,t2] * GᶜG[t1,t2]'
  ΣᵇG(t1,t2) = +2.0im * V₀^2 * GᶠG[t1,t2] * GᶜL[t1,t2]'

  # real-time collision integral. NOTE: @assert t1 >= t2
  ∫dt(i,j,A,B) = sign(j-i) * integrate(times[min(i,j):max(i,j)], 
    [A(t1,t) * B[t,t2] for t=min(i,j):max(i,j)], Trapezoidal())

  # conduction electron
  dGᶜL = -1.0im * (-0.0 * GᶜL[t1,t2] - 1.0im * σ * GᶜL[t1,t2])
  dGᶜG = -1.0im * (-0.0 * GᶜG[t1,t2] - 1.0im * σ * GᶜG[t1,t2])
  
  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f - λ₀) * GᶠL[t1,t2] + ∫dt(1,t1,ΣᶠG,GᶠL) - ∫dt(1,t2,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f - λ₀) * GᶠG[t1,t2] + ∫dt(t2,t1,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (-λ₀ * GᵇL[t1,t2] + ∫dt(1,t1,ΣᵇG,GᵇL) - ∫dt(1,t2,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (-λ₀ * GᵇG[t1,t2] + ∫dt(t2,t1,ΣᵇG,GᵇG))
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end

function f_diag_time(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG = data
  @unpack ω₀,λ,γ,V₀ = model

  ΣᶠL(t1,t2) = +1.0im * V₀^2 * GᵇL[t1,t2] * GᶜL[t1,t2]
  ΣᶠG(t1,t2) = +1.0im * V₀^2 * GᵇG[t1,t2] * GᶜG[t1,t2]
  ΣᵇL(t1,t2) = +2.0im * V₀^2 * GᶠL[t1,t2] * GᶜG[t1,t2]'
  ΣᵇG(t1,t2) = +2.0im * V₀^2 * GᶠG[t1,t2] * GᶜL[t1,t2]'

  # real-time collision integral
  ∫dt(A::GreenFunction,B::Function) = integrate(times[1:t1], [A[t1,t] * B(t,t1) for t=1:t1], Trapezoidal())
  ∫dt(A::Function,B::GreenFunction) = integrate(times[1:t1], [A(t1,t) * B[t,t1] for t=1:t1], Trapezoidal())

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
  
  return [dGᶜL, dGᶜG, dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end
