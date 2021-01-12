### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ f6eb14a0-1d04-11eb-21ab-5fe40ebb915d
begin
	using Revise
	using SpecialFunctions: erf, erfc
	using LinearAlgebra: diag, diagind, norm
	using Parameters: @unpack
	using Plots; Plots.gr(); #Plots.plotly()
	using LaTeXStrings
	using Test
end

# ╔═╡ b44ab15a-4d2e-11eb-16dc-21ca8a270562
html"""<style>
main {
    max-width: 1000px;
}
"""

# ╔═╡ 13bb1f84-1d06-11eb-306b-5bef219f7434
module M
	begin
		include("../src/model.jl")
		include("../src/util.jl")
	end
end 

# ╔═╡ 25249ec4-1d05-11eb-211f-17a1bb80cd03
model, data = let β=1e2,  # temperature
	nᵞ = 0,  # initial photon occupation
	tc = 120, # peak arrival time
	Ω = 0.2, # frequency bandwitdh of the pulse
	Γ = 20., # pump and loss strength
	
	# Pulse shape
	ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))

	model = M.PhotonAssitedModel(
		#= SIAM parameters =#
		V₀=sqrt(0.1),
		ε₀f=-0.5,
		σ=0.2,

		#= pulse parameters =#
		Γ₀=sqrt(0.05),  # light-matter coupling
		ω₀=0.01, # photon-ground state energy
		δ₀=Ω # photon artificial width
		# γ=t->Γ * ξ(t),
		# λ=t->Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2)))
	)

	data = M.ProblemData(model, β, M.NCA.LinearMesh(-15.0, 15.0, 2^15), (T -> 15ξ(T), τ -> 15ξ(tc+τ)))
	model, data
end;

# ╔═╡ 376c9bfc-1d05-11eb-2f4e-331c0302a14b
begin
	plot(data.mesh.mesh, data.GᵇT[1] |> imag;
		label=L"$\tilde{A}_b$")
	plot!(data.mesh.mesh, data.GᶠT[1] |> imag;
		label=L"$\tilde{A}_f$", legend=:topleft)
	plot!(twinx(), data.mesh.mesh, data.GᶜA[1] |> imag;
		label=L"$A_c$", linecolor=:green, grid=false, legend=:topright)
	title!(L"\textrm{Equilibrium spectral functions}")
	xlabel!(L"$\omega/D$")
	# savefig("equilibrium-spectra.pdf")
end

# ╔═╡ 1fb01056-1d06-11eb-241f-49b9b90929dd
begin
	# Initial condition
	u0 = [data.GᶠL,data.GᶠG,data.GᵇL,data.GᵇG]
	l0 = [data.GᶜA,data.GᶠT,data.GᶠM,data.GᵇT,data.GᵇM]
	
	# Vertical step
	f_vert(u, times, t1::Int, t2::Int) = M.f_vert_time(model, data, times, t1, t2)
	f_vert(u, v, times, t1::Int, t2::Int) = f_vert(u, times, t1, t2) .+ M.f_vert_freq(model, data, times, t1, t2)
	
	# Diagonal step
	f_diag(u, times, t1::Int) = (println("t: $(round(times[t1]; digits=3))"); M.f_diag_time(model, data, times, t1))
	f_diag(u, v, times, t1::Int) = f_diag(u, times, t1) .+ update_norms!(M.f_diag_freq(model, data, times, t1))
	
	# Vertical step for mixed functions
	f_line(u, v, times, t1::Int) = M.f_line_freq(model, data, times, t1)
	
	thermal_norms = float(Inf)
	update_norms!(n) = (global thermal_norms; thermal_norms = norm.(n); n)
	
	# Criteria for evaluating the thermal collision integrals
	# TODO: Accumulate norms over horizontal step (don't take into account just the diagonal)
	thermal_criteria(norms) = max(norms...) < 1e-7

	tspan = (0.0,200.0)
	
	# @time sol, caches = M.kbsolve(f_vert, f_diag, u0, tspan;
	# 	update_time =(_,t1,t2)->M.calculate_Σ_time!(model,data,t1,t2),
	# 	f_line=f_line, l0 = l0,
	# 	update_line =(_,t1)->M.calculate_Σ_freq!(model,data,t1),
	# 	# stop=()->thermal_criteria(thermal_norms),
	# 	dtini=1e-5, atol=1e-6, rtol=5e-4, dtmax=0.5)
	
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
	@time sol, caches = M.kbsolve(f_vert, f_diag, u0, tspan;
		atol=1e-6, rtol=1e-3, dtini=1e-8, dtmax=10.,
		update_time =(t,t1,t2)->M.calculate_Σ_time!(model,data,t,t1,t2))
	
	# Extract info
	@unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,GᵞL,GᵞG,ΣᶠG,ΣᵇG,ΣᶠL,ΣᵇL = data
	@unpack GᶜA,GᶠT,GᶠM,GᵇT,GᵇM,ΣᶠT,ΣᶠM,ΣᵇT,ΣᵇM,mesh,fermi,fermi⁻ = data
	times = sol.t;
end

# ╔═╡ 82b3677a-1d06-11eb-1a41-c3a9b29e9995
begin
	using Interpolations
	
	times_lin = range(first(times), stop=last(times), length=length(times))
	
	gd_itp = interpolate((times,times), +1.0im * (GᶠG.data .* conj(GᵇL.data) - GᶠL.data .* conj(GᵇG.data)), Gridded(Linear()))
	
	ρ, (x_, y_) = M.wigner_transform([gd_itp(t1,t2) for t1 in times_lin, t2 in times_lin]; 
		ts=times_lin, fourier=true)
	
	h1 = heatmap(y_, x_, -ρ |> real;
		xaxis=nothing, ylabel=L"\omega / D", ylims=(-2.5, 2.5), legend=:none, c=:magma)
	
	colorbar = scatter([0,0], [1,1], zcolor=[0,1];
		xaxis=nothing, xlims=(1,1), yaxis=nothing, label=nothing, grid=nothing, 
		c=:magma, clims=extrema(-ρ |> real))
	
	# Occupations
	p1 = plot(times, GᶠL.data |> imag ∘ diag;
		label=L"n_f")
	p1 = plot!(times, -GᵇL.data |> imag ∘ diag;
		label=L"n_b")
	p1 = plot!(times, (2GᶠL.data - GᵇL.data) |> imag ∘ diag;
		label=L"Q", line=(:dot, :grey, 1), legend=:topleft, xlims=extrema(times))
	
	# Kondo time-scale
	kondo_T = 1 / M.NCA.kondo_temp(data.mesh, vec(data.GᶜA.data) ./ (2pi) |> imag, model.ε₀f, model.V₀)
	
	# THz pulse
	pulse = [-GᵞL(t,t) for t in times] |> imag
	
	p1 = plot!(twinx(), times, pulse;
		label=L"n_\gamma", line=(:dot, :green), legend=:topright,
		xlabel=L"T\;D \quad(T_K\;D=%$(round(kondo_T, digits=3)))", xlims=extrema(times), grid=false)
		
	tK_grid = range(-1.0, 1.0, length=2) #NOTE: Needs an even number for a centred vspan
	
	for t in 2:length(tK_grid)		
		vspan!(times[findmax(pulse)[2]] .+ kondo_T .* [tK_grid[t-1], tK_grid[t]];
			alpha=0.5ℯ^(-abs(tK_grid[t])), fillcolor=:grey,
			linealpha=0.5ℯ^(-abs(tK_grid[t])), linecolor=:grey,
			markersize=0,
			label=nothing)
	end
	
	plot(h1, p1, colorbar, layout = @layout [grid(2, 1) a{0.05w}])
	# savefig("ad_wigner.pdf")
end

# ╔═╡ aede18f0-360f-11eb-182f-0fbb1762f733
# # Comparison of frequency solution with time solution
# begin
# 	ts = M.ft(mesh, zero(mesh.mesh), mode=-1)[1]
	
# 	x = hcat(data.GᵇT[1], data.GᶠT[1], data.GᶜA[1]) ./ (2pi)
# 	L = mapslices(x -> M.ft(mesh, x .* data.fermi, mode=-1)[2], x, dims=1) # Lesser
# 	G = mapslices(x -> M.ft(mesh, x .* data.fermi⁻, mode=-1)[2], x, dims=1) # Spectral
	
# 	τs = times - reverse(times)
	
# 	GbLτ = plot(ts, -L[:,1] |> imag;
# 		label=L"\mathcal{F}_\omega\{G^<_b\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GbLτ = plot!(τs, diag(reverse(GᵇL.data, dims=2)) |> imag;
# 		label=L"G^<_b(\tau)")
	
# 	GbGτ = plot(ts, -G[:,1] |> imag;
# 		label=L"\mathcal{F}_\omega\{G^>_b\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GbGτ = plot!(τs, diag(reverse(GᵇG.data, dims=2)) |> imag;
# 		label=L"G^>_b(\tau)")
	
# 	GfLτ = plot(ts, L[:,2] .|> imag;
# 		label=L"\mathcal{F}_\omega\{G^<_f\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GfLτ = plot!(τs, diag(reverse(GᶠL.data, dims=2)) |> imag;
# 		label=L"G^<_f(\tau)")
	
# 	GfGτ = plot(ts, -G[:,2] |> imag;
# 		label=L"\mathcal{F}_\omega\{G^>_f\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GfGτ = plot!(τs, diag(reverse(GᶠG.data, dims=2)) |> imag;
# 		label=L"G^>_f(\tau)")
	
# 	GcLτ = plot(ts, L[:,3] |> imag;
# 		label=L"\mathcal{F}_\omega\{G^<_c\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GcLτ = plot!(τs, GᶜL.(τs) |> imag;
# 		label=L"G^<_c(\tau)")
	
# 	GcGτ = plot(ts, -G[:,3] |> imag;
# 		label=L"\mathcal{F}_\omega\{G^>_c\}(\tau)", legend=:bottomright, xlims=(0.25τs[1], 0.25τs[end]), xlabel=L"\tau D")
# 	GcGτ = plot!(τs, GᶜG.(τs) |> imag;
# 		label=L"G^>_c(\tau)")
		
# 	plot(GbLτ, GbGτ, GfLτ, GfGτ, GcLτ, GcGτ, layout = @layout [a b; c d; e f])
# 	# savefig("equilibrium-comparison1.pdf")
# end

# ╔═╡ 21e84c2a-5407-11eb-3bbf-6590b0ca21eb
# begin
# 	mesh_real = M.NCA.LinearMesh(mesh.xmin, mesh.xmax, mesh.L)
# 	imGd = 1/(2pi) * (
# 		+ M.NCA.xcorr(mesh_real, fermi .* imag(GᵇT[1]), fermi⁻ .* imag(GᶠT[1]))
# 		+ M.NCA.xcorr(mesh_real, fermi⁻ .* imag(GᵇT[1]), fermi .* imag(GᶠT[1]))
# 		)
	
# 	Gd_τ = plot(ts, 1/(2pi) * M.ft(mesh_real, 1.0im * imGd; mode=-1)[2] |> imag;
# 		label=L"\mathcal{F}^{-1}_\omega\{A_d(\omega)\}(\tau)", 
# 		xlabel=L"\tau \;D", xlims=(0.25τs[1], 0.25τs[end]), 
# 		legend=:topright)
# 	Gd_τ = plot!(τs, 1im * diag(reverse(GᶠG.data .* conj(GᵇL.data) - GᶠL.data .* conj(GᵇG.data), dims=2)) |> imag;
# 		label=L"A_d(\tau)")
	
# 	τs_lin = range(τs[1], stop=τs[end], length=length(τs))
# 	ωs = M.ft(τs_lin, zero(τs_lin))[1]
	
# 	itp = interpolate((τs,), +1.0im * diag(reverse(GᶠG.data .* conj(GᵇL.data) - GᶠL.data .* conj(GᵇG.data), dims=2)),
# 		Gridded(Linear()))
	
# 	Gd_ω = plot(mesh.mesh, imGd; 
# 		label=L"A_d(\omega)", 
# 		xlabel=L"\omega / D", xlims=(ωs[1], ωs[end]),
# 		legend=:topright)
# 	Gd_ω = plot!(ωs, M.ft(τs_lin, itp.(τs_lin))[2] |> imag;
# 		label=L"\mathcal{F}_\tau\{A_d(\tau)\}(\omega)", 
# 	)
	
# 	plot(Gd_τ, Gd_ω, layout = grid(2,1))
# 	# savefig("equilibrium-comparison2.pdf")
# end

# ╔═╡ 19b432e6-1d14-11eb-1770-fda8ed6ad743
@testset "KMS boundary conditions" begin
	# G^R(t,t) = -i
	# @test all(diag(GᶜG.data - GᶜL.data) .≈ -1.0im)
	# @test all(diag(GᵞG.data - GᵞL.data) .≈ -1.0im)
	@test all(diag(GᶠG.data) .≈ -1.0im)
	@test all(diag(GᵇG.data) .≈ -1.0im)
	
	@test maximum(+imag(GᶠL.data)) <= 1.0 # Maximum occupation is 1
	@test maximum(-imag(GᵇL.data)) <= 1.0 # Maximum occupation is 1
	@test maximum(-imag(GᶠG.data)) <= 1.0
	@test maximum(-imag(GᵇG.data)) <= 1.0
	
	@test minimum(+imag(GᶠL.data)) >= -1.0
	@test minimum(-imag(GᵇL.data)) >= -1.0
	@test minimum(-imag(GᶠG.data)) >= -1.0
	@test minimum(-imag(GᵇG.data)) >= -1.0
	
# 	if iszero(model.Γ₀)
# 		∫dω(f) = M.integrate(freqs.mesh, f, M.TrapezoidalEvenFast()) / (2π)
# 		≅(a,b) = isapprox(a,b;atol=5e-3)

# 		# C^<(t,t0) = C˺(t,t0)
# 		@test all(GᶜL[t,1] ≅ +∫dω(fermi .* (-conj(Aᶜ˹[t]))) for t in 1:size(Aᶜ˹,2))
# 		@test all(GᶠL[t,1] ≅ +∫dω(fermi .* (-conj(GᶠT[t]))) for t in 1:size(GᶠT,2))
# 		@test all(GᵇL[t,1] ≅ -∫dω(fermi .* (-conj(GᵇT[t]))) for t in 1:size(GᶠT,2))
# 		@test all(ΣᶠL[t,1] ≅ +∫dω(fermi .* (-conj(ΣᶠT[t]))) for t in 1:size(ΣᶠT,2))
# 		@test all(ΣᵇL[t,1] ≅ -∫dω(fermi .* (-conj(ΣᵇT[t]))) for t in 1:size(ΣᵇT,2))

# 		# C^>(t0,t) = C˹(t0,t)
# 		@test all(GᶜG[1,t] ≅ -∫dω(fermi⁻ .* Aᶜ˹[t]) for t in 1:size(Aᶜ˹,2))
# 		@test all(GᶠG[1,t] ≅ -∫dω(fermi⁻ .* GᶠT[t]) for t in 1:size(GᶠT,2))
# 		@test all(GᵇG[1,t] ≅ -∫dω(fermi⁻ .* GᵇT[t]) for t in 1:size(GᵇT,2))
# 		@test all(ΣᶠG[1,t] ≅ -∫dω(fermi⁻ .* ΣᶠT[t]) for t in 1:size(ΣᶠT,2))
# 		@test all(ΣᵇG[1,t] ≅ -∫dω(fermi⁻ .* ΣᵇT[t]) for t in 1:size(ΣᵇT,2))
# 		else
# 		println("Cannot test mixed functions BCs")
# 	end
end

# ╔═╡ Cell order:
# ╠═b44ab15a-4d2e-11eb-16dc-21ca8a270562
# ╠═f6eb14a0-1d04-11eb-21ab-5fe40ebb915d
# ╠═13bb1f84-1d06-11eb-306b-5bef219f7434
# ╠═25249ec4-1d05-11eb-211f-17a1bb80cd03
# ╠═376c9bfc-1d05-11eb-2f4e-331c0302a14b
# ╠═1fb01056-1d06-11eb-241f-49b9b90929dd
# ╠═82b3677a-1d06-11eb-1a41-c3a9b29e9995
# ╠═aede18f0-360f-11eb-182f-0fbb1762f733
# ╠═21e84c2a-5407-11eb-3bbf-6590b0ca21eb
# ╠═19b432e6-1d14-11eb-1770-fda8ed6ad743
