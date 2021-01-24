Base.@kwdef mutable struct PhotonAssitedModel
  # single impurity Anderson model parameters
  V₀::Float64     # hybridization strength
  ε₀f::Float64    # f-electron ground state energy
  λ₀::Float64=NaN # gauge chemical potential
  σ::Float64      # c-electron gaussian bandwidth

  # pulse parameters
  Γ₀::Float64     # light-matter coupling strength
  ω₀::Float64     # light pulse energy
  Ω::Float64      # light pulse bandwidth
  tc::Float64     # light pulse peak time
  Π::Float64      # light pulse max occupation
end

struct ProblemData{T,U}
  GᶜL#::GreenFunction{T, Array{T,2}, Lesser}
  GᶜG#::GreenFunction{T, Array{T,2}, Greater}
  GᶠL::GreenFunction{T, Array{T,2}, Lesser}
  GᶠG::GreenFunction{T, Array{T,2}, Greater}
  GᵇL::GreenFunction{T, Array{T,2}, Lesser}
  GᵇG::GreenFunction{T, Array{T,2}, Greater}
  GᵞL₀
  GᵞG₀
  # GᵞL::GreenFunction{T, Array{T,2}, Lesser}
  # GᵞG::GreenFunction{T, Array{T,2}, Greater}
  
  ΣᶠL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᶠG::GreenFunction{T, Array{T,2}, Greater}
  ΣᵇL::GreenFunction{T, Array{T,2}, Lesser}
  ΣᵇG::GreenFunction{T, Array{T,2}, Greater}
  TᵞL::GreenFunction{T, Array{T,2}, Lesser}
  TᵞG::GreenFunction{T, Array{T,2}, Greater}

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

function ProblemData(model::PhotonAssitedModel, β::Real, mesh, nᵞ)
  # Get the correlated tilde and non-tilde Green functions
  imG_t, G = begin

    # Equilibrium model
    nca = FreqNCA(
      model=SIAM(
        V₀=model.V₀,
        ε₀f=model.ε₀f,
        tilde=true,
        Nc₀=x->2pi * NCA.gaussian(x, -0.0, model.σ)),
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

  ts, L_ = ft(nca.mesh, 1.0im/(2pi) * Nc .* fermi, mode=-1)
  ts, G_ = ft(nca.mesh, 1.0im/(2pi) * Nc .* fermi⁻, mode=-1)
  GᶜL = extrapolate(interpolate((ts,), +L_, Gridded(Linear())), 0)
  GᶜG = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)
  GᶜA = GreenFunction(ElasticArray(hcat(1.0im * Nc)), MixedGreater)

  GᶠL = GreenFunction(ones(1,1) * (+1.0im * ∫dω(fermi .* 2imG_t[:,2])), Lesser)
  GᶠG = GreenFunction(ones(1,1) * (-1.0im), Greater)
  GᶠT = GreenFunction(ElasticArray(hcat(2im * imG_t[:,2])), MixedGreater)
  GᶠM = GreenFunction(ElasticArray(hcat(G[:,2])), MixedGreater)

  GᵇL = GreenFunction(ones(1,1) * (-1.0im * ∫dω(fermi .* 2imG_t[:,1])), Lesser)  
  GᵇG = GreenFunction(ones(1,1) * (-1.0im), Greater)
  GᵇT = GreenFunction(ElasticArray(hcat(2im * imG_t[:,1])), MixedGreater)
  GᵇM = GreenFunction(ElasticArray(hcat(G[:,1])), MixedGreater)

  GᵞL₀(t,t′) = -1.0im * exp(-1.0im * (t-t′) * model.ω₀) * nᵞ(0.5(t+t′))
  GᵞG₀(t,t′) = -1.0im * exp(-1.0im * (t-t′) * model.ω₀) * (nᵞ(0.5(t+t′)) + 1)
  # GᵞL₀(t,t′) = -1.0im * (cos((t-t′) * model.ω₀) * (nᵞ(0.5(t+t′)) + 0.5)) + 0.5 * sin((t-t′) * model.ω₀)
  # GᵞG₀(t,t′) = -1.0im * (cos((t-t′) * model.ω₀) * (nᵞ(0.5(t+t′)) + 0.5)) - 0.5 * sin((t-t′) * model.ω₀)

  # @assert !iszero(nᵞ(0.0)) "The occupation at t=t0 has to be at least ϵ > 0"
  # GᵞL₀ = GreenFunction(ones(1,1) * (-im * nᵞ(0.0)), Lesser)
  # GᵞG₀ = GreenFunction(ones(1,1) * (-im * (nᵞ(0.0) + 1)), Greater)
  # GᵞL = GreenFunction(ones(1,1) * (-im * nᵞ(0.0)), Lesser)
  # GᵞG = GreenFunction(ones(1,1) * (-im * (nᵞ(0.0) + 1)), Greater)
  
  # Occupations report
  println("\nEquilibrium occupations:")
  println(" ∫dω f(ω) Aᶜ(ω) = $(imag(+GᶜL(0.0)))")
  println(" ∫dω f(ω) Aᶠ(ω) = $(imag(+GᶠL[1,1]))")
  println(" ∫dω b(ω) Aᵇ(ω) = $(imag(-GᵇL[1,1]))")
  println(" Q = $(imag(2GᶠL[1,1] - GᵇL[1,1]))")

  data = ProblemData{ComplexF64,Float64}(
    GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,#GᵞL,GᵞG
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    typeof(GᵇL)(zero(GᵇL.data)),
    typeof(GᵇG)(zero(GᵇG.data)),
    typeof(GᶠL)(zero(GᶠL.data)),
    typeof(GᶠG)(zero(GᶠG.data)),
    GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,
    typeof(GᶠT)(ElasticArray(zero(GᶠT.data))),
    typeof(GᶠM)(ElasticArray(zero(GᶠM.data))),
    typeof(GᵇT)(ElasticArray(zero(GᵇT.data))),
    typeof(GᵇM)(ElasticArray(zero(GᵇM.data))),
    mesh,fermi,fermi⁻,th½)

  @testset "Σ equivalence between mixed and equilibrium formulations" begin
    calculate_Σ_time!(model, data, [0.0], 1, 1)
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
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data
  @unpack ε₀f,λ₀ = model

  # real-time collision integral
  function ∫dt(i,j,A::GreenFunction,B::GreenFunction)
    retval = zero(eltype(A))
    @fastmath @simd for k in min(i,j):(max(i,j)-1)
        @inbounds retval += (times[k+1] - times[k]) * (A[t1,k]*B[k,t2] + A[t1,k+1]*B[k+1,t2])
    end
    return 1//2 * sign(j-i) * retval
  end
  
  # pseudo fermion
  dGᶠL = -1.0im * ((ε₀f + λ₀) * GᶠL[t1,t2] + ∫dt(1,t1,ΣᶠG,GᶠL) - ∫dt(1,t2,ΣᶠL,GᶠG))
  dGᶠG = -1.0im * ((ε₀f + λ₀) * GᶠG[t1,t2] + ∫dt(t2,t1,ΣᶠG,GᶠG))
  
  # slave boson
  dGᵇL = -1.0im * (λ₀ * GᵇL[t1,t2] + ∫dt(1,t1,ΣᵇG,GᵇL) - ∫dt(1,t2,ΣᵇL,GᵇG))
  dGᵇG = -1.0im * (λ₀ * GᵇG[t1,t2] + ∫dt(t2,t1,ΣᵇG,GᵇG))

  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]

  # thz pulse
  # function ∫dt(i,j,A::Function,B::GreenFunction)
  #   retval = zero(eltype(B))
  #   @fastmath @simd for k in min(i,j):(max(i,j)-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A(times[t1],times[k])*B[k,t2] + A(times[t1],times[k+1])*B[k+1,t2])
  #   end
  #   return 1//2 * sign(j-i) * retval
  # end
  # function ∫dt(i,j,A::GreenFunction,B::Function)
  #   retval = zero(eltype(A))
  #   @fastmath @simd for k in min(i,j):(max(i,j)-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A[t1,k]*B(times[k],times[t2]) + A[t1,k+1]*B(times[k+1],times[t2]))
  #   end
  #   return 1//2 * sign(j-i) * retval
  # end
  # @unpack GᵞL₀,GᵞG₀,TᵞL,TᵞG = data
  # @unpack ω₀,Ω,tc = model
  # dGᵞL = -1.0im * (ω₀ * GᵞL₀[t1,t2] - 1.0im * Ω^2/2 * (times[t1] - tc) * GᵞL₀[t1,t2] 
  #   + ∫dt(1,t1,TᵞG,GᵞL₀) - ∫dt(1,t2,TᵞL,GᵞG₀) - ∫dt(t2,t1,TᵞL,GᵞL₀))
  # dGᵞG = -1.0im * (ω₀ * GᵞL₀[t1,t2] - 1.0im * Ω^2/2 * (times[t1] - tc) * GᵞL₀[t1,t2]
  #   + ∫dt(1,t2,TᵞG,GᵞL₀) - ∫dt(1,t1,TᵞL,GᵞG₀) + ∫dt(t2,t1,TᵞG,GᵞG₀))

  # dGᵞL0 = -1.0im * (ω₀ * GᵞL₀[t1,t2] - 1.0im * Ω^2/2 * (times[t1] - tc) * GᵞL₀[t1,t2])
  # dGᵞG0 = -1.0im * (ω₀ * GᵞG₀[t1,t2] - 1.0im * Ω^2/2 * (times[t1] - tc) * GᵞL₀[t1,t2])

  # return [dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG, dGᵞL0, dGᵞG0]
end

function f_diag_time(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG = data

  # real-time collision integral
  function ∫dt(A::GreenFunction,B::GreenFunction)
    retval = zero(eltype(A))
    @fastmath @simd for k in 1:(t1-1)
        @inbounds retval += (times[k+1] - times[k]) * (A[t1,k]*B[k,t1] + A[t1,k+1]*B[k+1,t1])
    end
    return 1//2 * retval
  end

  # commutator
  ⋊(a,b) = a * b - b * adjoint(a)

  # pseudo fermion
  dGᶠL = -1.0im * (∫dt(ΣᶠG,GᶠL) - ∫dt(ΣᶠL,GᶠG) + ∫dt(GᶠL,ΣᶠG) - ∫dt(GᶠG,ΣᶠL))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (∫dt(ΣᵇG,GᵇL) - ∫dt(ΣᵇL,GᵇG) + ∫dt(GᵇL,ΣᵇG) - ∫dt(GᵇG,ΣᵇL))
  dGᵇG = 0.0
  
  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]

  # # thz pulse
  # @unpacl GᵞL₀,GᵞG₀,TᵞL,TᵞG = data
  # @unpack Ω,tc = model
  # function ∫dt(A::Function,B::GreenFunction)
  #   retval = zero(eltype(B))
  #   @fastmath @simd for k in 1:(t1-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A(times[t1],times[k])*B[k,t1] + A(times[t1],times[k+1])*B[k+1,t1])
  #   end
  #   return 1//2 * retval
  # end
  # function ∫dt(A::GreenFunction,B::Function)
  #   retval = zero(eltype(A))
  #   @fastmath @simd for k in 1:(t1-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A[t1,k]*B(times[k],times[t1]) + A[t1,k+1]*B(times[k+1],times[t1]))
  #   end
  #   return 1//2 * retval
  # end

  # I = ∫dt(TᵞG,GᵞL₀) - ∫dt(TᵞL,GᵞG₀) + ∫dt(GᵞL₀,TᵞG) - ∫dt(GᵞG₀,TᵞL)

  # T = times[t1]
  # dGᵞL = -1.0im * (-1.0im * (Ω^2 * (T - tc)) * GᵞL₀(T,T) + I)#(T < tc ? GᵞL₀(T,T) : data.GᵞL[t1,t1]) + I)
  # dGᵞG = -1.0im * (-1.0im * (Ω^2 * (T - tc)) * GᵞL₀(T,T) + I)#(T < tc ? GᵞL₀(T,T) : data.GᵞL[t1,t1]) + I)

  # I = ∫dt(TᵞG,GᵞL₀) - ∫dt(TᵞL,GᵞG₀) - ∫dt(GᵞL₀,TᵞG) - ∫dt(GᵞG₀,TᵞL)

  # dGᵞL = -1.0im * real(-1.0im * (Ω^2 * (times[t1] - tc)) * 0.0data.GᵞL₀[t1,t1] + I)
  # dGᵞG = -1.0im * real(-1.0im * (Ω^2 * (times[t1] - tc)) * 0.0data.GᵞL₀[t1,t1] + I)

  # dGᵞL0 = -1.0im * (-1.0im * (Ω^2 * (times[t1] - tc)) * GᵞL₀[t1,t1])
  # dGᵞG0 = -1.0im * (-1.0im * (Ω^2 * (times[t1] - tc)) * GᵞL₀[t1,t1])

  # return [dGᶠL, dGᶠG, dGᵇL, dGᵇG, dGᵞL, dGᵞG, dGᵞL0, dGᵞG0]
end

function f_vert_freq(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(mesh.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)
  
  # pseudo fermion
  dGᶠL = -1.0im * (
    +∫dω(conj(ΣᶠM[t1]) .* GᶠT[t2] - conj(ΣᶠT[t1]) .* GᶠM[t2]))
  dGᶠG = 0.0
  
  # slave boson
  dGᵇL = -1.0im * (
    -∫dω(conj(ΣᵇM[t1]) .* GᵇT[t2] - conj(ΣᵇT[t1]) .* GᵇM[t2]))
  dGᵇG = 0.0
  
  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end

function f_diag_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi = data

  # real-frequency collision integral
  ∫dω(f) = integrate(mesh.mesh, fermi .* f, TrapezoidalEvenFast()) / (2π)

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
  # # thz pulse
  # @unpacl GᵞL₀,GᵞG₀,TᵞL,TᵞG = data
  # @unpack Ω,tc = model
  # function ∫dt(A::Function,B::GreenFunction)
  #   retval = zero(eltype(B))
  #   @fastmath @simd for k in 1:(t1-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A(times[t1],times[k])*B[k,t1] + A(times[t1],times[k+1])*B[k+1,t1])
  #   end
  #   return 1//2 * retval
  # end
  # function ∫dt(A::GreenFunction,B::Function)
  #   retval = zero(eltype(A))
  #   @fastmath @simd for k in 1:(t1-1)
  #       @inbounds retval += (times[k+1] - times[k]) * (A[t1,k]*B(times[k],times[t1]) + A[t1,k+1]*B(times[k+1],times[t1]))
  #   end
  #   return 1//2 * retval
  # end

  return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
end

function f_line_freq(model::PhotonAssitedModel,data::ProblemData,times,t1)
  @unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG = data
  @unpack ε₀f,λ₀,σ = model

  # real-time collision integral
  ∫dt(A,B) = integrate(times, [A[t] .* B[t,t1] for t=1:t1], TrapezoidalFast())
  
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

function calculate_Σ_time!(model::PhotonAssitedModel,data::ProblemData,times,t1,t2)
  @unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL₀,GᵞG₀,ΣᶠL,ΣᶠG,ΣᵇL,ΣᵇG,TᵞL,TᵞG = data
  @unpack V₀,Γ₀ = model
  
  if (n = size(GᶠL,2)) > size(ΣᶠL,2)
    resize!(ΣᶠL,n)
    resize!(ΣᶠG,n)
    resize!(ΣᵇL,n)
    resize!(ΣᵇG,n)
    resize!(TᵞL,n)
    resize!(TᵞG,n)
  end

  V²L = V₀^2 + Γ₀^2 * 1.0im * (GᵞL₀(times[t1],times[t2]) + GᵞG₀(times[t2],times[t1]))
  V²G = V₀^2 + Γ₀^2 * 1.0im * (GᵞG₀(times[t1],times[t2]) + GᵞL₀(times[t2],times[t1]))
  # V²L = V₀^2 + Γ₀^2 * 1.0im * (GᵞL₀[t1,t2] + GᵞG₀[t2,t1])
  # V²G = V₀^2 + Γ₀^2 * 1.0im * (GᵞG₀[t1,t2] + GᵞL₀[t2,t1])

  ΣᶠL[t1,t2] = +1.0im * V²L * GᵇL[t1,t2] * GᶜL(times[t1] - times[t2])
  ΣᶠG[t1,t2] = +1.0im * V²G * GᵇG[t1,t2] * GᶜG(times[t1] - times[t2])
  ΣᵇL[t1,t2] = -2.0im * V²L * GᶠL[t1,t2] * GᶜG(times[t2] - times[t1])
  ΣᵇG[t1,t2] = -2.0im * V²G * GᶠG[t1,t2] * GᶜL(times[t2] - times[t1])
  TᵞG[t1,t2] = -Γ₀^2 * (
    + GᶠG[t1,t2] * GᵇL[t2,t1] * GᶜL(times[t2] - times[t1])
    + GᶠL[t2,t1] * GᵇG[t1,t2] * GᶜG(times[t1] - times[t2]))
  TᵞL[t1,t2] = -Γ₀^2 * (
    + GᶠL[t1,t2] * GᵇG[t2,t1] * GᶜG(times[t2] - times[t1])
    + GᶠG[t2,t1] * GᵇL[t1,t2] * GᶜL(times[t1] - times[t2]))
end

function calculate_Σ_freq!(model::PhotonAssitedModel,data::ProblemData,t1)
  @unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM = data
  @unpack mesh, fermi, fermi⁻, th½ = data
  @unpack V₀ = model
  
  if (n = size(GᶠT,2)) > size(ΣᶠT,2)
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
