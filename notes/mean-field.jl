### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ f2b6c08c-29ed-11eb-2363-6143e3b233a5
begin 
	using LinearAlgebra: tr, I
	using StaticArrays
	using Parameters: @with_kw
	
	using HCubature: hcubature
	∫dk(f; d=1) = hcubature(f, repeat([-π], d), repeat([π], d);
		atol=1e-8, rtol=1e-6)[1] / (2π)^d
	
	import NLsolve
	nlsolve(f, x0; ftol=1e-12) = NLsolve.nlsolve(f, x0; 
		method=:newton, autodiff=:forward, ftol=ftol, iterations=20,
		inplace=false, show_trace=true)
	
	using ForwardDiff: gradient
	using Test
end

# ╔═╡ 0cf46aec-25bd-11eb-0d6e-e99bd5e2acb6
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
	function H(k::SVector, p::PAM, b, λ)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = (p.V₀ * b)
		return [c h; h' f]
	end
	
	"""
	Slave-boson mean-field occupation of the fermionic fields
	"""
	function G0L(k::SVector, p::PAM, b, λ)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = (p.V₀ * b)

		σ = 0.5 * (c + f)
		δ = 0.5 * (c - f)
		Δ = sqrt(δ^2 + abs2(h))
		return [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
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

			return [σ - Δ, σ - Δ, σ + Δ, σ + Δ]
		end
		
		return (λ - p.ε₀f) * (abs2(b) - 1) + sum(∫dk(bands)[1:2])
	end
	
	"""
	Slave-boson mean-field equations at T -> 0
	"""
	function sbmf_saddle_points0(x::Vector, p::PAM)
		b, λ = x
		n = ∫dk(k-> G0L(k, p, b, λ))
		return [(λ - p.ε₀f) * b + p.V₀ * n[2,1], abs2(b) + n[2,2] - 1]
	end
	
	function sbmf_saddle_points1(x::Vector, p::PAM)
		return gradient(y -> FreeEnergy(y, p), x)
	end
end

# ╔═╡ 31fa9f1e-25c0-11eb-39ff-492bb3e7c011
begin
	# Parameters
	pam0 = PAM(V₀=1.0, tf=-0.0, ε₀f=-1.5)

	# Find the fields that minimize the Free Energy
	@show sol0 = nlsolve(x -> sbmf_saddle_points0(x, pam0), [0.2,1e-1])
	@show sol1 = nlsolve(x -> sbmf_saddle_points1(x, pam0), [0.2,1e-1])
	pam0.b, pam0.λ = sol0.zero
	
	@testset "Initial conditions" begin   
		n0 = ∫dk(k -> G0L(k, pam0, pam0.b, pam0.λ))

		# λ * b + V cf = 0
		@test ((pam0.λ - pam0.ε₀f) * pam0.b + pam0.V₀ * n0[2,1]) ≈ 0.0 atol=1e-7

		# f^2 + b^2 = 1
		@test abs2(pam0.b) + n0[2,2] - 1.0 ≈ 0.0 atol=1e-7

		# Half filling
		@test n0[1,1] + n0[2,2] ≈ 2
		
		# Equivalence of formulations
		@test sol0.zero ≈ sol1.zero
	end
end

# ╔═╡ 31f97be0-2a6e-11eb-3f71-1f370f449c07
begin
	using DifferentialEquations: ODEProblem, RK4, solve
	
	"""
	GL(t,t) Occupation of the quantum fields
	"""
	function GL(t, k::SVector, p::PAM, b, λ)
		T = promote(t, λ(t)) |> typeof ∘ first
		
		G0 = Complex{T}.(G0L(k, p, b(0.0), λ(0.0)))
		
		⋊(a,b) = a * b - b * adjoint(a)

		prob = ODEProblem(
			(u,_,t) -> -1.0im * (H(k,p,b(t),λ(t)) ⋊ u), 
			G0,
			T.((0.0, t))
		)
		return solve(prob, RK4(), reltol=1e-6, abstol=1e-8)[end]
	end
	
	# Equivalence of formulations
	@test ∫dk(k -> 
		GL(0.0, k, pam0, t -> pam0.b, t -> pam0.λ) - G0L(k, pam0, pam0.b, pam0.λ)) |> iszero
	
	# Test automatic diff of GL(t)
	@show nlsolve([0.8]) do λ
		real(∫dk(k -> GL(1.0, k, pam0, t -> pam0.b, t -> λ[1]))[2,2]) + abs2(pam0.b) - 1
	end
end

# ╔═╡ fcd5b4fa-2a4d-11eb-2b43-79699425e293
begin
	using DifferentialEquations: DDEProblem, DDEFunction, MethodOfSteps, QBDF, Tsit5
	
	# Photon occupation
	function ξ(t; Ω=2.0,tc=5)
	  return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
	end
	
	# Photon-matter interaction strength
	const Γ = sqrt(0.2)
	
	"""
	Slave boson mean-field equations in time
	"""
	function sbmf(u, h, p, t)
		println(" t: $t")
		
		b = u[1] + 1.0im * u[2]
		λ = u[3]
		
		# Sum over all momenta
		n = ∫dk(k -> begin
			g = GL(t, k, p[1], t -> h(p,t)[1] + 1.0im * h(p,t)[2], t -> h(p,t)[3])
			α = g[2,1] * b' + g[1,2] * b
			dαdb = g[2,1] + p[1].V₀ * (b * g[1,1] * g[2,2] + b' * g[2,1]^2)
			dαdλ = g[2,2] * (g[1,2] * b + g[2,1] * b')
			SVector{5}(g[2,2], g[2,1], α, dαdb, dαdλ)
		end)
		
		db = -1.0im * ((λ - p[1].ε₀f) * b + p[1].V₀ * n[2])
		db+= -1.0im * (2Γ^2 * ξ(t) * n[3] * n[4])
		dλ = abs2(b) + n[1] - 1
		dλ+= 2Γ^2 * ξ(t) * n[3] * n[5]
		
		return [real(db), imag(db), real(dλ)]
	end
	
	# History function ({b(t), λ(t)} for t < 0)
	h(p,t) = [real(pam0.b), imag(pam0.b), real(pam0.λ)]
	
	# Initial condition
	u0 = [real(pam0.b), imag(pam0.b), real(pam0.λ)]
	
	# Differential-Algebraic Delay Equation
	f = DDEFunction(sbmf, mass_matrix=[1 0 0; 0 1 0; 0 0 0])
	prob = DDEProblem(f, u0, h, (0.0, 10.0), (pam0,))
	sol = solve(prob, MethodOfSteps(QBDF()))
end

# ╔═╡ Cell order:
# ╠═f2b6c08c-29ed-11eb-2363-6143e3b233a5
# ╠═0cf46aec-25bd-11eb-0d6e-e99bd5e2acb6
# ╠═31fa9f1e-25c0-11eb-39ff-492bb3e7c011
# ╠═31f97be0-2a6e-11eb-3f71-1f370f449c07
# ╠═fcd5b4fa-2a4d-11eb-2b43-79699425e293
