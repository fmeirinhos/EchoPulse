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
	# default(fmt = :png)
end

# ╔═╡ 13bb1f84-1d06-11eb-306b-5bef219f7434
module M include("../src/model.jl") end

# ╔═╡ 25249ec4-1d05-11eb-211f-17a1bb80cd03
model, data = let β=5e-1,  # temperature
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
		σ=0.3,

		#= pulse parameters =#
		Γ₀=0.0,  # light-matter coupling
		ω₀=0.0, # photon-ground state energy
		δ₀=1e-1, # photon artificial width
		γ=t->0.0,#Γ * ξ(t),
		λ=t->0.0#Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2)))
	)

	data = M.ProblemData(model, β, M.NCA.LinearMesh(-15.0, 15.0, 2^14), nᵞ)
	model, data
end;

# ╔═╡ 376c9bfc-1d05-11eb-2f4e-331c0302a14b
begin
	plot(data.mesh.mesh, data.GᵇT[1] |> imag, label=L"$A_b$")
	plot!(data.mesh.mesh, data.GᶠT[1] |> imag, label=L"$A_f$")
	plot!(data.mesh.mesh, data.GᶜA[1] |> imag, label=L"$A_c$")
	title!("Tilde spectral functions")
	xlabel!(L"$\omega$")
end

# ╔═╡ 1fb01056-1d06-11eb-241f-49b9b90929dd
begin
	# Initial condition
	u0 = [data.GᶜL,data.GᶜG,data.GᶠL,data.GᶠG,data.GᵇL,data.GᵇG,data.GᵞL,data.GᵞG]
	l0 = [data.GᶜA,data.GᶠT,data.GᶠM,data.GᵇT,data.GᵇM]
	
	# Vertical step
	f_vert(u, times, t1::Int, t2::Int) = M.f_vert_time(model, data, times, t1, t2)
	f_vert(u, v, times, t1::Int, t2::Int) = f_vert(u, times, t1, t2) .+ M.f_vert_freq(model, data, times, t1, t2)
	
	# Diagonal step
	f_diag(u, times, t1::Int) = (println("t: $(times[t1])"); M.f_diag_time(model, data, times, t1))
	f_diag(u, v, times, t1::Int) = f_diag(u, times, t1) .+ update_norms!(M.f_diag_freq(model, data, times, t1))
	
	# Vertical step for mixed functions
	f_line(u, v, times, t1::Int) = M.f_line_freq(model, data, times, t1)
	
	thermal_norms = float(Inf)
	update_norms!(n) = (global thermal_norms; thermal_norms = norm.(n); n)
	
	# Criteria for evaluating the thermal collision integrals
	# TODO: Accumulate norms over horizontal step (don't take into account just the diagonal)
	thermal_criteria(norms) = max(norms...) < 1e-7

	tspan = (0.0,100.0)
end

# ╔═╡ 5c4e194a-1d06-11eb-02fd-49d757a246b6
begin
	# sol, caches = M.kbsolve(f_vert, f_diag, u0, (tspan[1], tspan[2]);
	# 	update_time =(_,t1,t2)->M.calculate_Σ_time!(model,data,t1,t2),
	# 	update_line =(_,t1)->M.calculate_Σ_freq!(model,data,t1),
	# 	f_line=f_line, l0 = l0,
	# 	stop=()->thermal_criteria(thermal_norms),
	# 	dtini=1e-3, atol=5e-6, rtol=5e-4, dtmax=0.5)
	
	# If we don't need to evaluate the thermal collision integrals anymore
	# if thermal_criteria(thermal_norms)
	#   # We need a new integrator-state without the mixed functions
	#   state = KadanoffBaym.VCABMState((state.u[1],), state.t, state.dt)
	#   state, caches = kbsolve(f_vert, f_diag, u0[1], tspan[1], tspan[2];
	# 	atol=1e-7, rtol=1e-5, max_dt=1.0,
	# 	w_time! = (t1,t2) -> calculate_Σ_time!(model,data,t1,t2),
	# 	st_ch = (state, caches))
	# end

	# The solver will stop when tspan[2] is reached or `stop` evaluates to `false`
	sol, caches = M.kbsolve(f_vert, f_diag, u0, tspan;
	    update_time =(_,t1,t2)->M.calculate_Σ_time!(model,data,t1,t2),
		atol=1e-6, rtol=1e-4, dtmax=1.0)
	
	# Extract info
	@unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG,ΣᶠL,ΣᵇL = data
	@unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi,fermi⁻ = data
	times = sol.t;
end;

# ╔═╡ effe5844-27ec-11eb-2a5b-37d2d3d4cef5
# Comparison with equilibrium solution
begin
	using KadanoffBaym.FFTW
	
	ωs = mesh
	dω= ωs[2] - ωs[1]
	
	ts = fftfreq(length(ωs), 2π/dω)
	is = sortperm(ts)
	
	ℯⁱᵠ = dω * exp.(1.0im * ωs[1] .* ts)
	
	x = hcat(0.5 * data.GᵇT[1], 0.5 * data.GᶠT[1], 0.5 * data.GᶜA[1]) ./ pi
	L = mapslices(x -> ℯⁱᵠ .* fft(x .* data.fermi), x, dims=1)  # Lesser
	G = mapslices(x -> ℯⁱᵠ .* fft(x .* data.fermi⁻), x, dims=1) # Spectral
	
	τs = range(-last(sol.t), stop=last(sol.t), length=length(sol.t))
end

# ╔═╡ d292740a-268b-11eb-2fff-1be92be92655
begin
	plot(ts[is], -L[is,1] |> imag, label="G_b Lesser equi")
	plot!(τs, diag(reverse(GᵇL.data, dims=2)) |> imag, label="G_b Lesser time")
	xlims!(0.25τs[1], 0.25τs[end])
end

# ╔═╡ b1b158e0-27ef-11eb-0296-c9ff28a7a375
begin
	plot(ts[is], -G[is,1] |> imag, label="G_b Greater equi")
	plot!(τs, diag(reverse(GᵇG.data, dims=2)) |> imag, label="G_b Greater time")
	xlims!(0.25τs[1], 0.25τs[end])
end

# ╔═╡ 38d6f8c0-27f0-11eb-16ba-8155267c0470
begin
	plot(ts[is], L[is,2] |> imag, label="G_f Lesser equi")
	plot!(τs, diag(reverse(GᶠL.data, dims=2)) |> imag, label="G_f Lesser time")
	xlims!(0.25τs[1], 0.25τs[end])
end

# ╔═╡ 6da30e7a-268a-11eb-3a63-27618476574a
begin
	plot(ts[is], -G[is,2] |> imag, label="G_f Greater equi")
	plot!(τs, diag(reverse(GᶠG.data, dims=2)) |> imag, label="G_f Greater time")
	xlims!(0.25τs[1], 0.25τs[end])
end

# ╔═╡ e7107556-27f0-11eb-1f4f-e7f0d36c418e
begin
	plot(ts[is], L[is,3] |> imag, label="G_c Lesser equi")
	plot!(τs, diag(reverse(GᶜL.data, dims=2)) |> imag, label="G_c Lesser time")
	xlims!(τs[1], τs[end])
end

# ╔═╡ f15d3b8e-27f0-11eb-018a-b76fa4c9202f
begin
	plot(ts[is], -G[is,3] |> imag, label="G_c Greater equi")
	plot!(τs, diag(reverse(GᶜG.data, dims=2)) |> imag, label="G_c Greater time")
	xlims!(τs[1], τs[end])
end

# ╔═╡ 82b3677a-1d06-11eb-1a41-c3a9b29e9995
begin
	plot(times, [imag(GᶠL[t,t]) for t in eachindex(times)], label=L"f")
	plot!(times, [imag(-GᵇL[t,t]) for t in eachindex(times)], label=L"b")
	plot!(twinx(), times, imag([2GᶠL[t,t] - GᵇL[t,t] for t in eachindex(times)]), line=:dot, color=:green, legend=:right, label=L"Q")#, ylims=(1-1e-6, 1+1e-6))
	xlabel!(L"t")
	ylabel!(L"n")
end

# ╔═╡ 19b432e6-1d14-11eb-1770-fda8ed6ad743
@testset "KMS boundary conditions" begin
  # G^R(t,t) = -i
  @test all(diag(GᶜG.data - GᶜL.data) .≈ -1.0im)
  @test all(diag(GᵞG.data - GᵞL.data) .≈ -1.0im)
  @test all(diag(GᶠG.data) .≈ -1.0im)
  @test all(diag(GᵇG.data) .≈ -1.0im)

#   if iszero(model.Γ₀)
# 	∫dω(f) = M.integrate(freqs.mesh, f, M.TrapezoidalEvenFast()) / (2π)
# 	≅(a,b) = isapprox(a,b;atol=5e-3)

# 	# C^<(t,t0) = C˺(t,t0)
# 	@test all(GᶜL[t,1] ≅ +∫dω(fermi .* (-conj(Aᶜ˹[t]))) for t in 1:size(Aᶜ˹,2))
# 	@test all(GᶠL[t,1] ≅ +∫dω(fermi .* (-conj(GᶠT[t]))) for t in 1:size(GᶠT,2))
# 	@test all(GᵇL[t,1] ≅ -∫dω(fermi .* (-conj(GᵇT[t]))) for t in 1:size(GᶠT,2))
# 	@test all(ΣᶠL[t,1] ≅ +∫dω(fermi .* (-conj(ΣᶠT[t]))) for t in 1:size(ΣᶠT,2))
# 	@test all(ΣᵇL[t,1] ≅ -∫dω(fermi .* (-conj(ΣᵇT[t]))) for t in 1:size(ΣᵇT,2))

# 	# C^>(t0,t) = C˹(t0,t)
# 	@test all(GᶜG[1,t] ≅ -∫dω(fermi⁻ .* Aᶜ˹[t]) for t in 1:size(Aᶜ˹,2))
# 	@test all(GᶠG[1,t] ≅ -∫dω(fermi⁻ .* GᶠT[t]) for t in 1:size(GᶠT,2))
# 	@test all(GᵇG[1,t] ≅ -∫dω(fermi⁻ .* GᵇT[t]) for t in 1:size(GᵇT,2))
# 	@test all(ΣᶠG[1,t] ≅ -∫dω(fermi⁻ .* ΣᶠT[t]) for t in 1:size(ΣᶠT,2))
# 	@test all(ΣᵇG[1,t] ≅ -∫dω(fermi⁻ .* ΣᵇT[t]) for t in 1:size(ΣᵇT,2))
#   else
# 	println("Cannot test mixed functions BCs")
#   end
	
	@test maximum(+imag(GᶠL.data)) <= 1.0 # Maximum occupation is 1
	@test maximum(-imag(GᵇL.data)) <= 1.0 # Maximum occupation is 1
	@test maximum(-imag(GᶠG.data)) <= 1.0
	@test maximum(-imag(GᵇG.data)) <= 1.0
	
	# @test minimum(+imag(GᶠL.data)) >= 0.0
	# @test minimum(-imag(GᵇL.data)) >= 0.0
	# @test minimum(-imag(GᶠG.data)) >= 0.0
	# @test minimum(-imag(GᵇG.data)) >= 0.0
end

# ╔═╡ 606fba54-1d12-11eb-166f-6ba2ed4920ca
# begin
# 	plot(times, GᵞL.data |> ((-) ∘ diag ∘ imag), label="Lesser", xlabel=L"t")
# 	plot!(times, GᵞG.data |> ((-) ∘ diag ∘ imag), label="Greater")
# 	plot!(twinx(), times, model.ξ.(times), color=:green, legend=:right, label="Pulse", line=:dot)
# end

# ╔═╡ 49ce0562-1d17-11eb-04c8-1369eb331816
# heatmap(times, times, ΣᶠL.data[1:length(times),1:length(times)] |> imag)
# heatmap(times, times, GᶠG.data[1:length(times),1:length(times)] |> imag)

# ╔═╡ f07d9a18-1d16-11eb-10f4-87e5112010a7
plot(times[1:end-1], times[2:end] - times[1:end-1], label=L"dt")

# ╔═╡ Cell order:
# ╠═a3f2e1bc-1eae-11eb-31d8-2b565c9a452e
# ╠═13bb1f84-1d06-11eb-306b-5bef219f7434
# ╠═f6eb14a0-1d04-11eb-21ab-5fe40ebb915d
# ╠═25249ec4-1d05-11eb-211f-17a1bb80cd03
# ╠═376c9bfc-1d05-11eb-2f4e-331c0302a14b
# ╠═1fb01056-1d06-11eb-241f-49b9b90929dd
# ╠═5c4e194a-1d06-11eb-02fd-49d757a246b6
# ╠═effe5844-27ec-11eb-2a5b-37d2d3d4cef5
# ╠═d292740a-268b-11eb-2fff-1be92be92655
# ╠═b1b158e0-27ef-11eb-0296-c9ff28a7a375
# ╠═38d6f8c0-27f0-11eb-16ba-8155267c0470
# ╠═6da30e7a-268a-11eb-3a63-27618476574a
# ╠═e7107556-27f0-11eb-1f4f-e7f0d36c418e
# ╠═f15d3b8e-27f0-11eb-018a-b76fa4c9202f
# ╠═82b3677a-1d06-11eb-1a41-c3a9b29e9995
# ╠═19b432e6-1d14-11eb-1770-fda8ed6ad743
# ╠═606fba54-1d12-11eb-166f-6ba2ed4920ca
# ╠═49ce0562-1d17-11eb-04c8-1369eb331816
# ╠═f07d9a18-1d16-11eb-10f4-87e5112010a7
