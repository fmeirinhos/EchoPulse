### A Pluto.jl notebook ###
# v0.19.6

using Markdown
using InteractiveUtils

# ╔═╡ f2b6c08c-29ed-11eb-2363-6143e3b233a5
begin
	using Pkg; Pkg.activate()

	using StaticArrays
	using GenericLinearAlgebra, LinearAlgebra

	using ForwardDiff

	import NLsolve
	nlsolve(f, x0; ftol=1e-12) = NLsolve.nlsolve(f, x0; 
		method=:newton, autodiff=:forward, 
		ftol=ftol, iterations=20,
		inplace=false, show_trace=true)

	using Interpolations
	using DifferentialEquations

	using Plots, LaTeXStrings
end

# ╔═╡ 25366804-7db3-4672-8c80-2d762e6b6675
begin
	using FFTW, KadanoffBaym
	using QuadGK, SpecialFunctions
	using HCubature: hcubature

	dim = 3
	N = 2^16

	rtol=1e-6
	atol=1e-10

	# k-integral
	∫dk(f) = hcubature(f, repeat([-π], dim), repeat([π], dim); rtol=rtol, atol=atol)[1] / (2π)^dim

	dos = let
		ωs = collect(range(-2dim, +2dim, length=N-1))
		ts = KadanoffBaym.ft(ωs, ωs)[1]

		(ωs, ρs) = KadanoffBaym.ft(ts, besselj0.(2ts) .^ dim; inverse=true)

		interpolate((collect(ωs), ), real(ρs), Gridded(Linear()))
	end

	# dos-integral
	∫dε(f) = quadgk(ε -> f(ε) * dos(ε), -2dim, +2dim, rtol=rtol, atol=atol)[1]
end

# ╔═╡ ba6ea9f1-2d8f-4853-b433-9825aef236a9
begin
	"""
	Cubic Anderson model parameters (units of tc)
	"""
	Base.@kwdef mutable struct PAM
		V0::Float64         # Hybridization strength
		tf::Float64         # F-electron hopping
		ε0f::Float64        # F-electron ground state e
		μ::Float64 = 0.0    # Chemical potential

		Γ0::Float64 = 0.0   # Photon hybridization strength
		ω0::Float64 = 0.0   # Photon frequency
		Ω0::Float64 = 0.0   # Photon width
		tc::Float64 = 0.0   # Photon incidence time
	end

	"""
	Aux-boson mean-field Hamiltonian
	"""
	function H(k::SVector, p::PAM, b, λ; t=0.0)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = p.V0 * b
		return @SMatrix [c h'; h f]
	end

	function H(ε::Float64, p::PAM, b, λ; t=0.0)
		c = ε - p.μ
		f = p.tf * abs2(b) * ε + λ - p.μ
		h = p.V0 * b
		return @SMatrix [c h'; h f]
	end

	"""
	Effective action at T -> 0
	"""
	function Seff(x::Vector, p::PAM)
		b, λ = x

		# The factor of 2 accounts for spin-degeneracy
		return (λ - p.ε0f) * (abs2(b) - 1) + 2∫dε(ε -> begin
			εs = eigvals(Hermitian(H(ε, p, b, λ)))
			sum(εi for εi in εs if εi < λ)
			end
		)
	end

	"""
	Saddle-point equations at T -> 0
	"""
	function saddle_points(x::Vector, p::PAM)
		return ForwardDiff.gradient(y -> Seff(y, p), x)
	end
end;

# ╔═╡ 31fa9f1e-25c0-11eb-39ff-492bb3e7c011
begin
	# Model
	pam = PAM(V0=1.0, tf=-0.0, ε0f=-1.5)

	# Find the fields that minimize the Free Energy
	@time sol = nlsolve(x -> saddle_points(x, pam), [2.5e-1, 2.0e-1])
	@info sol
end;

# ╔═╡ 31f97be0-2a6e-11eb-3f71-1f370f449c07
begin
	"""
	Auxiliary-boson saddle-point occupation of the fermionic fields
	"""
	function GL0(ε::Float64, p::PAM, b::Number, λ::Number)
	    c = ε - p.μ
	    f = p.tf * abs2(b) * ε + λ - p.μ
	    h = (p.V0 * b)

	    σ = 0.5 * (c + f)
	    δ = 0.5 * (c - f)
	    Δ = sqrt(δ^2 + abs2(h))
	    return +im * @SMatrix [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
	end
	function GL0(k::SVector, p::PAM, b::Number, λ::Number)
	    c = -2.0 * sum(cos, k) - p.μ
	    f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
	    h = (p.V0 * b)

	    σ = 0.5 * (c + f)
	    δ = 0.5 * (c - f)
	    Δ = sqrt(δ^2 + abs2(h))
	    return +im * @SMatrix [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
	end

	"""
	G<(t,t) of the matter fields
	"""
	function GL(t::Number, ε::Float64, p::PAM, b, λ)
	    T = promote_type(typeof(t), typeof(λ(t)))
	    
	    u0 = GL0(ε, p, b(0.0), λ(0.0))
	    
	    if iszero(t)
	        return u0
	    else
	        return solve(
	            ODEProblem(
	                (u,_,t) -> (o = -1.0im * H(ε,p,b(t),λ(t);t=t) * u; o - o'), 
	                u0,
	                T.((0.0, t - eps(t)))
	                ), Tsit5(), reltol=1e-6, abstol=1e-10)[end]
	    end
	end
end;

# ╔═╡ 777ae10b-bfac-4f08-bc85-c256b2165823
begin
	# Gaussian shape
	function ξ(t; Ω)
	  return (Ω^2 / (2π))^(1/4) * exp(-t^2 * (Ω^2 / 4))
	end

	# Photon Green function
	GLξ(t, t′; ω, Ω, tc) = -im * exp(-im * (t - t′) * ω) * ξ(t - t′,Ω=Ω)^2 * (Ω^2 / (2π))^(-1/2) * ξ(0.5 * (t + t′) - tc,Ω=Ω)^2
	GGξ(t, t′; ω, Ω, tc) = -im * exp(-im * (t - t′) * ω) * ξ(t - t′,Ω=Ω)^2 * (Ω^2 / (2π))^(-1/2) * (1. + ξ(0.5 * (t + t′) - tc,Ω=Ω)^2)
	
    """
    Slave boson mean-field equations in time
    """
    function sbmf(u, p, t)
        @show t

        b = interpolate(
            (vcat(knots_, t), ), 
            vcat(u1_coefs + im * u2_coefs, u[1] + im * u[2]), 
            Gridded(Linear())
        )

        λ = interpolate(
            (vcat(knots_, t), ), 
            vcat(u3_coefs, u[3]), 
            Gridded(Linear())
        )

        gγ = -im * GLξ(t,t; ω=pam.ω0, Ω=pam.Ω0, tc=pam.tc)^2

        # Sum over all momenta    
#         n = ∫dε(ε -> begin
#             g = GL(t, ε, pam, b, λ)

#             α = g[1,2] * b(t)' + g[2,1] * b(t)

#             dαdb = g[1,2] + 0.5 * (pam.V0 * g[1,1]) * (g[2,2] - im) * b(t) + 0.5 * (pam.V0 * g[2,1]) * g[1,2]' * b(t)'
#             dαdλ = 0.5 * g[2,1] * (g[2,2] - im) * b(t) + 0.5 * g[2,2] * g[1,2]' * b(t)'

#             @SVector [g[1,2], g[2,2], ε * g[2,2], α, dαdb, dαdλ]
#         end)
        n = ∫dε(ε -> begin
            g = GL(t, ε, pam, b, λ)

            α = g[2,1] * b(t) + g[1,2] * b(t)' 

            dαdb = g[1,2] + 0.5 * (pam.V0 * g[1,1]) * g[2,2] * b(t) + 0.5 * (pam.V0 * g[1,2]) * g[1,2] * b(t)'
            dαdλ = 0.5 * g[2,1] * g[2,2] * b(t) + 0.5 * g[2,2] * g[1,2] * b(t)'

            @SVector [g[1,2], g[2,2], ε * g[2,2], α, dαdb, dαdλ]
        end)

        db = -im * ((λ(t) - pam.ε0f) * b(t) - im * (pam.V0 * n[1] - 2pam.tf * b(t) * n[3]))
        db+= -im * 2pam.Γ0^2 * gγ * n[4] * n[5]

        dλ = abs2(b(t)) - im * n[2] - 1
        dλ+= 2pam.Γ0^2 * gγ * n[4] * n[6]

        return [real(db), imag(db), real(dλ)]
    end

    knots_ = [-1.0,]

    u1_coefs = [real(sol.zero[1]), ]
    u2_coefs = [imag(sol.zero[1]), ]
    u3_coefs = [real(sol.zero[2]), ]

    cb = FunctionCallingCallback((u,t,integrator)-> begin
        push!(knots_, t)

        push!(u1_coefs, u[1])
        push!(u2_coefs, u[2])
        push!(u3_coefs, u[3])
    end)

    # Initial condition
    u0 = [real(sol.zero[1]), imag(sol.zero[1]), real(sol.zero[2])]

	pam.Γ0=0.5; pam.ω0=0.1; pam.Ω0=1.0; pam.tc=5.0
	
    # Differential-Algebraic Delay Equation
    f = ODEFunction(sbmf, mass_matrix=[1 0 0; 0 1 0; 0 0 0])
    prob = ODEProblem(f, u0, (0.0, 10.0))
    sol_t = solve(prob, QBDF(; autodiff=false), reltol=1e-4, abstol=1e-7)
end;

# ╔═╡ d543a368-c527-445c-999f-04ffcc048a83
begin
	plot(sol_t.t, (u -> sqrt(u[1]^2 + u[2]^2)).(sol_t.u), label=L"b", color=:darkblue)
	plot!(sol_t.t, (u -> u[3]).(sol_t.u), label=L"\lambda", color=:darkgreen)
	
	# plot!(twinx(), sol_t.t, t -> ξ(t)^2, label="photon-field", color=:darkred, legend=false)
end

# ╔═╡ e0a5ddb9-8c1a-46b7-a632-545c3f69e3fc
# function GL_photon(tmax::Number, b, λ; Ω=2.0, tc=10)
    
#     h(t) = 0.5im * Ω^2 * (tc - t)
    
#     v(t) = ∫dε(ε -> begin
#             g = GL(t, ε, pam, b, λ)
#             @SVector [g[2,1]^2 * (b(t)')^2, b(t) * b(t) * g[1,2]^2, 2abs2(b(t)) * g[1,1] * g[2,2]]
#         end)
        
#     solve(
#         ODEProblem(
#             (u,_,t) -> begin
#                 @show t
#                 flush(stdout)
#                 v_ = v(t)
#                 @show v_
#                 o = -1.0im * (h(t) + sum(v_)) * u
#                 o - o'
#             end, 
#             GLξ(0.0, 0.0),
#             (0.0, tmax)
#             ), Tsit5(), reltol=1e-6, abstol=1e-10, dtmax=1.0)
# end

# ╔═╡ 044abdb0-a0c9-4c01-8c69-919ea9602b2a
# gl = GL_photon(10, t -> sol_time[end](t)[1] + im * sol_time[end](t)[2], t -> sol_time[end](t)[3]);

# ╔═╡ b7147585-ee6d-4474-9ade-b9917401c76d
# begin
# 	function GL(ts::Vector, k::SVector, p::PAM, b, λ)
# 	    t0 = first(ts)
	    
# 	    T = promote_type(typeof(t0), typeof(λ(t0)))
	    
# 	    u0 = GL0(k, p, b(t0), λ(t0))
	    
# 	    u0 = solve(
# 	        ODEProblem(
# 	            (u,_,t) -> -1.0im * H(k,p,b(t),λ(t);t=t) * u, 
# 	            u0,
# 	            (t0, last(ts))
# 	        ), Tsit5(), reltol=1e-8, abstol=1e-10, saveat=ts)
	    
# 	    u0 = solve(
# 	        ODEProblem(
# 	            (u,_,t) -> [+1.0im * H(k,p,b(t),λ(t);t=t) * v for v in u], 
# 	            u0.u,
# 	            (t0, last(ts))
# 	        ), Tsit5(), reltol=1e-8, abstol=1e-10, saveat=ts)
	    
# 	    return hcat(u0.u...)
# 	end

# 	function GG(ts::Vector, k::SVector, p::PAM, b, λ)
# 	    t0 = first(ts)
	    
# 	    T = promote_type(typeof(t0), typeof(λ(t0)))
	    
# 	    u0 = GL0(k, p, b(t0), λ(t0)) - im * I
	    
# 	    u0 = solve(
# 	        ODEProblem(
# 	            (u,_,t) -> -1.0im * H(k,p,b(t),λ(t);t=t) * u, 
# 	            u0,
# 	            (t0, last(ts))
# 	        ), Tsit5(), reltol=1e-8, abstol=1e-10, saveat=ts)
	    
# 	    u0 = solve(
# 	        ODEProblem(
# 	            (u,_,t) -> [+1.0im * H(k,p,b(t),λ(t);t=t) * v for v in u], 
# 	            u0.u,
# 	            (t0, last(ts))
# 	        ), Tsit5(), reltol=1e-8, abstol=1e-10, saveat=ts)
	    
# 	    return hcat(u0.u...)
# 	end

# 	GbL = -im * [b_sol(t1)' * b_sol(t2) for t1 in ts, t2 in ts];
# 	GbG = -im * [b_sol(t1) * b_sol(t2)' for t1 in ts, t2 in ts];
# end

# ╔═╡ Cell order:
# ╠═f2b6c08c-29ed-11eb-2363-6143e3b233a5
# ╠═25366804-7db3-4672-8c80-2d762e6b6675
# ╠═ba6ea9f1-2d8f-4853-b433-9825aef236a9
# ╠═31fa9f1e-25c0-11eb-39ff-492bb3e7c011
# ╠═31f97be0-2a6e-11eb-3f71-1f370f449c07
# ╠═777ae10b-bfac-4f08-bc85-c256b2165823
# ╠═d543a368-c527-445c-999f-04ffcc048a83
# ╟─e0a5ddb9-8c1a-46b7-a632-545c3f69e3fc
# ╟─044abdb0-a0c9-4c01-8c69-919ea9602b2a
# ╟─b7147585-ee6d-4474-9ade-b9917401c76d
