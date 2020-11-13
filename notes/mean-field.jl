### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 0cf46aec-25bd-11eb-0d6e-e99bd5e2acb6
begin
	using LinearAlgebra: tr, I
	using StaticArrays
	using Parameters
	using Test
	
	using HCubature: hcubature
	∫dk(f; d=2) = hcubature(f, repeat([-π], d), repeat([π], d);
		atol=1e-6, rtol=1e-3)[1] / (2π)^d
	
	import NLsolve
	nlsolve(f, x0) = NLsolve.nlsolve(f, x0; 
		inplace=false, show_trace=true, method=:anderson, ftol=1e-6, iterations=20)
	
	using ForwardDiff: gradient
	saddle_points(f, x0) = nlsolve(x -> gradient(f, x), x0)
	
	using DifferentialEquations: ODEProblem, DDEProblem, MethodOfSteps, VCABM, Tsit5, solve, SavingCallback, SavedValues
	
	using Roots: find_zero
	using Interpolations
end

# ╔═╡ ae00abba-25bf-11eb-3b2d-c15bede300e7
begin
	"""
	Cubic Anderson model parameters (units of tc)
	"""
	@with_kw mutable struct PAM
	  V₀::Float64 = 0.    # Hybridization strength
	  tf::Float64 = 0.    # F-electron hopping
	  ε₀f::Float64 = 0.   # F-electron ground state e
	  μ::Float64 = 0.     # Chemical potential
	  β::Float64 = Inf    # Inverse temperature

	  b::Float64 = NaN    # Saddle-point condensed slave boson
	  λ::Float64 = NaN    # Saddle-point lagrange multiplier
	end
	
	"""
	Slave-boson mean-field Hamiltonian
	"""
	function H(k::SVector, p::PAM, b::Number=p.b, λ::Number=p.λ)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = (p.V₀ * b)
		return SMatrix{2,2,ComplexF64}(c, h, h', f)
	end
	
	"""
	Slave-boson mean-field occupation of the fermionic fields
	"""
	function G0L(k::SVector, p::PAM, b::Number=p.b, λ::Number=p.λ)
	  c = -2.0 * sum(cos, k) - p.μ
	  f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
	  h = (p.V₀ * b)

	  σ = 0.5 * (c + f)
	  δ = 0.5 * (c - f)
	  Δ = sqrt(δ^2 + abs2(h))
	  return SMatrix{2,2,ComplexF64}(1 - δ/Δ, -h'/Δ, -h/Δ, 1 + δ/Δ)
	end
	
	"""
	Slave-boson mean-field free energy at T -> 0
	"""
	function FreeEnergy(x::Vector, p::PAM)
	  b, λ = x

	  function bands(k::SVector)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = (p.V₀ * b)

		σ = 0.5 * (c + f)
		Δ = sqrt((0.5 * (c - f))^2 + abs2(h))

		return σ - Δ, σ - Δ, σ + Δ, σ + Δ
	  end

	  return (λ - p.ε₀f) * (abs2(b) - 1) + ∫dk(k -> sum(bands(k)[1:2]))
	end

	"""
	Slave-boson mean-field equations at T -> 0
	"""
	function sbmf_equations(x::Vector, p::PAM)
	  b, λ = x
	  G0 = ∫dk(k-> G0L(k, p, b, λ))
	  return [(λ - p.ε₀f) * b + p.V₀ * G0[2,1], abs2(b) + G0[2,2] - 1]
	end
end

# ╔═╡ b2cddc12-25bf-11eb-3f44-c1e241f80b55
begin
	# Parameters
	pam0 = PAM(V₀=1.0, tf=-0.0, ε₀f=-1.5)

	# Find the fields that minimize the Free Energy
	# @show solution = saddle_points(x -> FreeEnergy(x, pam_0), [0.2,1e-1])
	@show solution = nlsolve(x -> sbmf_equations(x, pam0), [0.2,1e-1])
	pam0.b, pam0.λ = solution.zero
end

# ╔═╡ 31fa9f1e-25c0-11eb-39ff-492bb3e7c011
@testset "Initial conditions" begin
  G0L_ = ∫dk(k -> G0L(k, pam0, pam0.b, pam0.λ))
  
  # λ * b + V cf = 0
  @test G0L_[1,2] * pam0.V₀ + (pam0.λ - pam0.ε₀f) * pam0.b ≈ 0.0 atol=1e-5
  
  # f^2 + b^2 = 1
  @test G0L_[2,2] + pam0.b^2 ≈ 1.0 atol=1e-5
  
  # Half filling
  @test G0L_[1,1] + G0L_[2,2] ≈ 2
end

# ╔═╡ 02348ffc-25c3-11eb-099e-152227b90de2
begin
	"""
	GL(t,t) Occupation of the fermionic fields
	"""
	function GL(t, k::SVector, b, λ)::SMatrix{2,2,ComplexF64}	
		prob = ODEProblem(
			(u,_,t) -> -1.0im * H(k,pam0,b(t),λ(t)) * u, G0L(k, pam0), (0.0, t))
		return solve(prob, Tsit5(), reltol=1e-3, abstol=1e-6)[end]
	end
	
	"""
	Alternative formulation for GL(t,t) iff [H(t), H(t′)] = 0
	"""
# 	function GL′(t, k::SVector, b, λ)::SMatrix{2,2,ComplexF64}
# 		using QuadGK
# 		∫dt(t0,tf,f) = quadgk(f, t0, tf; atol=1e-6, rtol=1e-3)[1]
		
# 		x = ∫dt(0, t, t′ -> [b(t′), abs2(b(t′)), λ(t′)])
# 		c = (-2.0 * sum(cos, k) - pam0.μ) * t
# 		f = -2.0pam0.tf * sum(cos, k) * x[2] - pam0.μ * t + x[3] 
# 		h = pam0.V₀ * x[1]
# 		U = exp(-1.0im * SMatrix{2,2,ComplexF64}(c, h, h', f))
# 		return U * G0L(k, pam0) * U'
# 	end
	
	# Photon occupation
	function ξ(t; Ω=1,tc=5)
	  return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4)) * 1/(exp(10 * (t - tc)) + 1)
	end
	
	# Light-matter coupling
	const Γ = sqrt(0.2)
end

# ╔═╡ 7ce0af3a-25c4-11eb-1815-999f35d3c92c
function f(u, h, p, t)
	println(" t: $t")
	
	b = t′ -> h(p, t′)
	λ = p[2]

	# solve for λ
	λ.coefs[end] = find_zero(λ′ -> begin
		# change data in the λ interpolator
		λ.knots[1][end] = t
		λ.coefs[end] = λ′

		z = ∫dk(k -> begin
		  g = GL(t, k, b, λ)
		  α = (g[1,2] * b(t)' + g[2,1] * b(t))
		  dαdλ = g[2,2] * α
		  SVector{3,ComplexF64}(g[2,2], α, dαdλ)
		end)
		
		real(abs2(b(t)) + z[1] - 1 + 2Γ^2 * ξ(t) * z[2] * z[3]) |> println
		real(abs2(b(t)) + z[1] - 1 + 2Γ^2 * ξ(t) * z[2] * z[3])
		end, λ.coefs[end-1], atol=1-5)
	
	z::SVector{3,ComplexF64} = ∫dk(k -> begin
		g = GL(t, k, b, λ)
		α = (g[1,2] * b(t)' + g[2,1] * b(t))
		dαdb = g[1,2] + pam0.V₀ * (g[1,1] * g[2,2] * b(t)' + g[2,1] * g[2,1] * b(t))
		SVector{3,ComplexF64}(g[2,1], α, dαdb)
	end)
	
	return -1.0im * ((λ(t) - pam0.ε₀f) * b(t) + pam0.V₀ * z[1] + 2Γ^2 * ξ(t) * z[2] * z[3])
end

# ╔═╡ 2d4d1282-25c5-11eb-2d18-61888b2aebe8
begin
	# Because λ is found through an implicit equation (and not as the result of the time integration) we need to make some tricks to properly save and access this function at all `t`s

	# In order to interpolate some function we need at least two points, hence we add an artificial past point
	λ = interpolate(([-2.0, -1.0],), [pam0.λ,pam0.λ], Gridded(Linear()))

	# Since the ODE solver is adaptive we should only store the points accepted by the integrator. This can be done with a `SavingCallback` which will add a new node to the interpolator's data
	cb = SavingCallback((u,t,integrator)-> begin
		push!(λ.knots[1], t)
		# push!(λ.coefs, NaN)
		push!(λ.coefs, λ.coefs[end] + 5e-2)
		u
		end, SavedValues(Float64, ComplexF64))
	
	# Initial condition
	u0 = complex(pam0.b)
	
	# History functio (b(t) for t <0)
	h(p,t) = complex(pam0.b)
	
	prob = DDEProblem(f, u0, h, (0.0, 1.0), (pam0, λ))
	sol = solve(prob, MethodOfSteps(VCABM()); callback=cb)
end

# ╔═╡ Cell order:
# ╠═0cf46aec-25bd-11eb-0d6e-e99bd5e2acb6
# ╠═ae00abba-25bf-11eb-3b2d-c15bede300e7
# ╠═b2cddc12-25bf-11eb-3f44-c1e241f80b55
# ╠═31fa9f1e-25c0-11eb-39ff-492bb3e7c011
# ╠═02348ffc-25c3-11eb-099e-152227b90de2
# ╠═7ce0af3a-25c4-11eb-1815-999f35d3c92c
# ╠═2d4d1282-25c5-11eb-2d18-61888b2aebe8
