### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ a3f2e1bc-1eae-11eb-31d8-2b565c9a452e
using Revise

# ╔═╡ f6eb14a0-1d04-11eb-21ab-5fe40ebb915d
begin
	using SpecialFunctions: erf, erfc
	using LinearAlgebra: diag, diagind, norm
	using Test
	using Parameters
	using Plots, LaTeXStrings
end

# ╔═╡ 13bb1f84-1d06-11eb-306b-5bef219f7434
module M include("../src/model-time.jl") end

# ╔═╡ 25249ec4-1d05-11eb-211f-17a1bb80cd03
model, data = let β=5e1,  # temperature
	nᵞ = 0,  # initial photon occupation
	tc = 20, # peak arrival time
	Ω = 1e0, # frequency bandwitdh of the pulse
	Γ = 10., # pump and loss strength
	
	# Pulse shape
	ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))

	model = M.PhotonAssitedModel(
		#= SIAM parameters =#
		V₀=sqrt(0.5),
		ε₀f=-0.5,
		σ=0.2,

		#= pulse parameters =#
		Γ₀=0.0,  # light-matter coupling
		ω₀=0.0, # photon-ground state energy
		δ₀=1e-1, # photon artificial width
		γ=t->0.0,#Γ * ξ(t),
		λ=t->0.0#Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2)))
	)

	data = M.ProblemData(model, β, M.NCA.LinearMesh(-10.0, 10.0, 2^11), nᵞ)
	model, data
end;

# ╔═╡ f0172c84-33ea-11eb-3760-d98cfb0c7ba5
begin
	# Initial condition
	u0 = [data.GᶠL,data.GᶠG,data.GᵇL,data.GᵇG]
	
	# Vertical step
	f_vert(u, times, t1::Int, t2::Int) = M.f_vert_time(model, data, times, t1, t2)
	
	# Diagonal step
	f_diag(u, times, t1::Int) = (println("t: $(times[t1])"); M.f_diag_time(model, data, times, t1))

	tspan = (data.ts,10.0)
	
	@time sol, caches = M.kbsolve(f_vert, f_diag, u0, tspan;
		atol=1e-6, rtol=1e-3, dtmax=1.0, dtini=1e-8)
	
	# Extract info
	@unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠG,ΣᵇG,ΣᶠL,ΣᵇL = data
	times = sol.t;
end

# ╔═╡ 82b3677a-1d06-11eb-1a41-c3a9b29e9995
begin
	plot(times, [imag(GᶠL[t,t]) for t in eachindex(times)], label=L"f")
	plot!(times, [imag(-GᵇL[t,t]) for t in eachindex(times)], label=L"b")
	plot!(twinx(), times, imag([2GᶠL[t,t] - GᵇL[t,t] for t in eachindex(times)]), line=:dot, color=:green, legend=:right, label=L"Q", ylims=(1-1e-6, 1+1e-6))
	xlabel!(L"t")
	ylabel!(L"n")
end

# ╔═╡ f07d9a18-1d16-11eb-10f4-87e5112010a7
plot(times[1:end-1], times[2:end] - times[1:end-1], label=L"dt")

# ╔═╡ Cell order:
# ╠═a3f2e1bc-1eae-11eb-31d8-2b565c9a452e
# ╠═13bb1f84-1d06-11eb-306b-5bef219f7434
# ╠═f6eb14a0-1d04-11eb-21ab-5fe40ebb915d
# ╠═25249ec4-1d05-11eb-211f-17a1bb80cd03
# ╠═f0172c84-33ea-11eb-3760-d98cfb0c7ba5
# ╠═82b3677a-1d06-11eb-1a41-c3a9b29e9995
# ╠═f07d9a18-1d16-11eb-10f4-87e5112010a7
