### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 13bb1f84-1d06-11eb-306b-5bef219f7434
module M
	begin
		using NLsolve, DSP, NCA
		using DSP.FFTW, Interpolations, KadanoffBaym
		using NumericalIntegration, ElasticArrays
		using Parameters: @unpack
		using Test
		include("../src/model.jl")
		using DSP.FFTW: fftfreq
		include("../src/util.jl")
	end
end

# ╔═╡ 3d95b35a-5e76-11eb-3118-61d944fb7c9a
begin
	using LinearAlgebra: diag, norm
	using Plots, LaTeXStrings
	Plots.gr()
end;

# ╔═╡ 25249ec4-1d05-11eb-211f-17a1bb80cd03
# This controls all the parameters of our model
model, data, time_span = begin 
	β  = 1.0e3  # temperature
	tc = 8.0e1  # frequency bandwitdh of the gaussian pulse
	Ω  = 2e-1 	# peak arrival time
	Π  = 1.0e1  # maximum photon occupation
	
	model = M.PhotonAssitedModel(
		#= SIAM parameters =#
		V₀=sqrt(0.08),	# hybridization strength
		ε₀f=-0.5,		# d-electron energy
		σ=0.45,			# gaussian cdos width

		#= quantum light parameters =#
		Γ₀=sqrt(0.01),  # light-matter coupling
		ω₀=1e0, 		# photon energy
		Ω=Ω,			# frequency bandwitdh of the gaussian pulse
		tc=tc,			# peak arrival time
		Π=Π 			# maximum photon occupation
	)
	
	# Pulse shape s.t. ∫dt ξ(t)^2 = 1
	ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
	
	data = M.ProblemData(
		model, 
		β, 
		M.NCA.LinearMesh(-20.0, 20.0, 2^16),  
		T -> (Π / Ω * sqrt(2pi)) * ξ(T)^2)
	
	time_span = (0.0, 250.0)
	
	model, data, time_span
end;

# ╔═╡ dfded9cc-5efe-11eb-387d-a35f79859dd8
begin
	cecu6_bandwidth = 2.0# eV
	cdos_bandwidth = 6model.σ # 3σ
	to_eV = cecu6_bandwidth / cdos_bandwidth
	fwmh_gaussian = 2 * sqrt(2 * log(2))
	fwmh_lorentzi = 2
	ω_to_eV(ω) = round(ω * to_eV , digits=3)
	ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2, digits=3)
	ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2), digits=4)
	ω_to_K(ω) = round(ω * to_eV * 1.1604518e4, digits=3)
	
	# Kondo frequency
	ωK = M.NCA.kondo_temp(data.mesh, vec(data.GᶜA.data) ./ (2pi) |> imag, model.ε₀f, model.V₀)
	
md"
#### Summary
Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $cecu6_bandwidth ``eV``, the chosen SIAM parameters were
1. ``\varepsilon_d =`` $(ω_to_eV(model.ε₀f)) ``eV`` (``d-``orbital energy)
2. ``V_0 =  `` $(ω_to_eV(model.V₀)) ``eV`` (hybridisation strength)
3. ``T_K = `` $(ω_to_K(ωK)) ``K`` (Kondo temperature: $(ω_to_ps(ωK)) ``ps``)
4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
and the gaussian-shaped pulse parameters were
5. ``\omega = `` $(ω_to_THz(model.ω₀)) ``THz`` (pulse energy)
6. ``\Omega = `` $(ω_to_THz(Ω)) ``THz``  (pulse bandwidth: $(6ω_to_ps(Ω)) ``ps``)
7. ``\Gamma = `` $(ω_to_eV(model.Γ₀)) ``eV`` (light-matter coupling)
8. ``\Pi = `` $(round(Π, digits=3)) ``n_\gamma`` (maximum photon-field occupation)

The system will be integrated from $(ω_to_ps(1/time_span[1])) to $(ω_to_ps(1/time_span[2])) ps with the light-pulse reaching its maximum at $(ω_to_ps(1/tc)) ps.

###### Questions
1. Do we need to shift the photons energy by the chemical potential?
2. How to convert maximum photon occupation to ``MV cm^{-1}``
3. Fix angular frequency problems: ``2\pi \omega t``
"
end

# ╔═╡ 376c9bfc-1d05-11eb-2f4e-331c0302a14b
begin
	imGd = 1/(2pi) * real(
		+ M.NCA.xcorr(data.mesh, data.fermi .* imag(data.GᵇT[1]), data.fermi⁻ .* imag(data.GᶠT[1]))
		+ M.NCA.xcorr(data.mesh, data.fermi⁻ .* imag(data.GᵇT[1]), data.fermi .* imag(data.GᶠT[1]))
		)
	
	plot(ω_to_eV.(data.mesh.mesh), data.GᶜA[1] |> imag;
		label=L"$A_c$", 
		linecolor=:green,
		legend=:topleft,
		xlabel=L"\omega \quad (eV)",#L"$\omega/D$",
		xlims=ω_to_eV.((-5.0, 5.0)),
		title=L"\textrm{Equilibrium spectral functions}"
		)
	
	plot!(ω_to_eV.(data.mesh.mesh), imGd,
		label=L"$A_d$",
		linecolor=:darkorange)
	
	vline!(ω_to_eV.([model.ε₀f]);
		label=L"\varepsilon_d",
		line=(:dot, 1.0, :blue, 0.5))
	
	vspan!(ω_to_eV.([-ωK/2, ωK/2]);
		fillcolor=:grey,
		alpha=0.3,
		label=L"\omega_K")
end

# ╔═╡ 1fb01056-1d06-11eb-241f-49b9b90929dd
# This controls the integration
begin
	@time sol, caches = M.kbsolve(
		(_,t,t1,t2) -> M.f_vert_time(model, data, t, t1, t2), 
		(_,t,t1) -> M.f_diag_time(model, data, t, t1), 
		[data.GᶠL,data.GᶠG,data.GᵇL,data.GᵇG],# data.GᵞL,data.GᵞG,data.GᵞL₀,data.GᵞG₀], 
		time_span;
		atol=1e-6, rtol=1e-3, dtini=1e-6, dtmax=10.,
		update_time =(t,t1,t2)->M.calculate_Σ_time!(model,data,t,t1,t2))

	# Extract info
	M.@unpack GᶜL,GᶜG,GᶠL,GᶠG,GᵇL,GᵇG,ΣᶠG,ΣᵇG,ΣᶠL,ΣᵇL = data
	times = sol.t
end;

# ╔═╡ 834f9d50-60c9-11eb-3207-3bb56de82801
begin
	function integrate(x::AbstractVector, y::AbstractVector)
		if isone(length(x))
			return zero(first(y))
		end
		@inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
		@inbounds @fastmath @simd for i in 2:(length(y) - 1)
			retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
		end
		return 1//2 * retval
	end
	
	struct TimeOrdered end
	struct AntiTimeOrdered end
	
	struct OrderedGF{T}
		L
		G
	end

	function Base.getindex(G::OrderedGF{TimeOrdered}, t1::Int, t2::Int)
		t1 > t2 ? G.G[t1,t2] : G.L[t1,t2]
	end
	function Base.getindex(G::OrderedGF{AntiTimeOrdered}, t1::Int, t2::Int)
		t1 >= t2 ? G.L[t1,t2] : G.G[t1,t2]
	end
	Base.getindex(G::OrderedGF, t1::Int, ::Colon) = [G[t1,t] for t in 1:size(G.L,2)]
	Base.getindex(G::OrderedGF, ::Colon, t2::Int) = [G[t,t2] for t in 1:size(G.L,1)]
	
	const TO = OrderedGF{TimeOrdered}
	const ATO = OrderedGF{AntiTimeOrdered}
	
	function dyson_lesser1(GL, GG, TL, TG)
		G = copy(GL)
		
		for t in eachindex(times)			
			G[t,t] += integrate(times, [TO(GL,GG)[t,t′] * (
						+ integrate(times, TO(TL,TG)[t′,:] .* GL[:,t])
						- integrate(times, TL[t′,:] .* ATO(GL,GG)[:,t])
							) for t′ in eachindex(times)])
			
			G[t,t] -= integrate(times, [GL[t,t′] * (
						+ integrate(times, TG[t′,:] .* GL[:,t])
						- integrate(times, ATO(TL,TG)[t′,:] .* ATO(GL,GG)[:,t])
							) for t′ in eachindex(times)])
		end
		
		return diag(G)
	end
	
	function dyson_lesser2(g0L, g0G, tL, tG)
		g = copy(g0L)

		for t1 in eachindex(times)
			t2 = t1

			function ∫dt(i,j,A,B::Function)
				return sign(j-i) * integrate(times[min(i,j):max(i,j)], 
					[A[t1,t] * B(t) for t=min(i,j):max(i,j)])
			end

			function ∫dt(t,i,j,A,B)
				retval = zero(eltype(A))
				@fastmath @simd for k in min(i,j):(max(i,j)-1)
					@inbounds retval += (times[k+1] - times[k]) * (A[t,k]*B[k,t2] + A[t,k+1]*B[k+1,t2])
				end
				return 1//2 * sign(j-i) * retval
			end

			g[t1,t2] += ∫dt(1, t1, g0G, t -> begin 
				∫dt(t,1,t,tG,g0L) + 
				∫dt(t,t,t2,tL,g0L) +
				∫dt(t,t2,1,tL,g0G)
			end)
			g[t1,t2] += ∫dt(t2, 1, g0L, t -> begin
				∫dt(t,1,t2,tG,g0L) + 
				∫dt(t,t2,t,tG,g0G) +
				∫dt(t,t,1,tL,g0G)
			end)
		end
		return g
	end
	
	# NOTE: Still have to be debuged!
	function dyson_lesser3(G0L, G0G, TL, TG)
		GL = copy(G0L)

		for t in eachindex(times)
			GL[t,t] += integrate(times, [G0G[t,t′] * (
						+ integrate(times[1:t′], TG[t′,1:t′] .* G0L[1:t′,t])
						+ integrate(times[t′:t], TL[t′,t′:t] .* G0L[t′:t,t])
						- integrate(times, TL[t′,:] .* G0G[:,t])
							) for t′ in eachindex(times)])

			GL[t,t] -= integrate(times, [G0L[t,t′] * (
						+ integrate(times, TG[t′,:] .* G0L[:,t])
						- integrate(times[t′:t], TG[t′,t′:t] .* G0G[t′:t,t])
						- integrate(times[1:t′], TL[t′,1:t′] .* G0G[1:t′,t])
							) for t′ in eachindex(times)])
		end
		return GL
	end
end;

# ╔═╡ 82b3677a-1d06-11eb-1a41-c3a9b29e9995
begin
	# Spectral function
	ρt = +1.0im * (GᶠG.data .* conj(GᵇL.data) - GᶠL.data .* conj(GᵇG.data))
	ρτ, (τs, Ts) = M.wigner_transform_itp(ρt, times; fourier=false)
	ρω, (ωs, Ts) = M.wigner_transform_itp(ρt, times; fourier=true)
	
	h1 = heatmap(ω_to_ps.(inv.(Ts)), ω_to_eV.(ωs), -ρω |> real;
		xaxis=nothing,
		ylabel=L"\omega \quad(eV)",#L"\omega / D",
		# ylims=(-2.5, 2.5),
		legend=:none,
		c=:magma,
		title=L"A_d(\omega,T)")
	
	colorbar = scatter([0,0], [1,1], zcolor=[0,1];
		xaxis=nothing,
		xlims=(1,1),
		yaxis=nothing,
		label=nothing,
		grid=nothing, 
		c=:magma,
		clims=extrema(-ρω |> real))
	
	# Occupations
	p1 = plot(ω_to_ps.(inv.(times)), GᶠL.data |> imag ∘ diag;
		label=L"n_f",
		title=L"\textrm{Field occupations}")
	p1 = plot!(ω_to_ps.(inv.(times)), -GᵇL.data |> imag ∘ diag;
		label=L"n_b")
	p1 = plot!(ω_to_ps.(inv.(times)), (2GᶠL.data - GᵇL.data) |> imag ∘ diag;
		label=L"Q",
		line=(:dot, :grey, 1),
		legend=:topleft,
		xlims=ω_to_ps.(inv.(times)))
	
	plt = twinx()
	
	# THz pulse
	G0L = [data.GᵞL₀(t,t′) for t in times, t′ in times]
	G0G = [data.GᵞG₀(t,t′) for t in times, t′ in times]
	
	TL = data.TᵞL.data[1:length(times),1:length(times)]
	TG = data.TᵞG.data[1:length(times),1:length(times)]
	
	GL = dyson_lesser1(G0L, G0G, TL, TG)
	
	p1 = plot!(plt, ω_to_ps.(inv.(times)), G0L |> (-) ∘ imag ∘ diag;
		label=nothing,
		line=(:green, 3.0, 0.2),
		legend=:topright,
		xlabel=L"T \quad (ps)",#L"T\;D",
		xlims=ω_to_ps.(inv.(times)),
		grid=false)
	p1 = plot!(plt, ω_to_ps.(inv.(times)), GL |> (-) ∘ imag;
		label=L"n_\gamma",
		line=(:green))
	
	# Kondo time-scale
	t_max_pulse = times[findmax(G0L |> (-) ∘ imag ∘ diag)[2]]
	vspan!(plt, ω_to_ps.(inv.(t_max_pulse .+ 1/ωK .* [-0.5, +0.5]));
		fillcolor=:grey,
		alpha=0.3,
		label=L"T_K")
	
	plot(h1, p1, colorbar, layout = @layout [grid(2, 1) a{0.05w}])
	# savefig("ad_wigner.pdf")
end

# ╔═╡ aede18f0-360f-11eb-182f-0fbb1762f733
# # Comparison of frequency solution with time solution
# # NOTE: Make sure that the light-matter coupling is zero!
# begin
# 	ts = M.ft(mesh, zero(mesh.mesh), mode=-1)[1]
	
# 	x = hcat(data.GᵇT[1], data.GᶠT[1], data.GᶜA[1]) ./ (2pi)
# 	L = mapslices(x -> M.ft(mesh, x .* data.fermi, mode=-1)[2], x, dims=1) # Lesser
# 	G = mapslices(x -> M.ft(mesh, x .* data.fermi⁻, mode=-1)[2], x, dims=1) # Spectral
		
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
# # Comparison of frequency *projected* solution with time solution
# # NOTE: Make sure that the light-matter coupling is zero!
# begin
# 	Gd_τ = plot(ts, 1/(2pi) * M.ft(mesh.mesh, 1.0im * imGd; mode=-1)[2] |> imag;
# 		label=L"\mathcal{F}^{-1}_\omega\{A_d(\omega)\}(\tau)", 
# 		xlabel=L"\tau \;D", 
# 		xlims=(0.25τs[1], 0.25τs[end]), 
# 		legend=:topright,
# 		alpha=1.0, ms=1.0)
	
# 	Gd_τ = plot!(τs, reverse(ρt, dims=2) |> diag ∘ imag;
# 		label=L"A_d(\tau)", ms=2.0)
		
# 	Gd_ω = plot(mesh.mesh, imGd; 
# 		label=L"A_d(\omega)", 
# 		xlabel=L"\omega / D", xlims=(ωs[1], ωs[end]),
# 		legend=:topright)
	
# 	Gd_ω = plot!(ωs, -ρω[:,length(times) ÷ 2] |> real;
# 		label=L"\mathcal{F}_\tau\{A_d(\tau)\}(\omega)", 
# 	)
	
# 	plot(Gd_τ, Gd_ω, layout = grid(2,1))
# 	# savefig("equilibrium-comparison2.pdf")
# end

# ╔═╡ 19b432e6-1d14-11eb-1770-fda8ed6ad743
# Kubo-Martin-Schwinger boundary conditions
M.@testset "KMS boundary conditions" begin
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
end;

# ╔═╡ Cell order:
# ╟─13bb1f84-1d06-11eb-306b-5bef219f7434
# ╟─3d95b35a-5e76-11eb-3118-61d944fb7c9a
# ╠═25249ec4-1d05-11eb-211f-17a1bb80cd03
# ╟─dfded9cc-5efe-11eb-387d-a35f79859dd8
# ╟─376c9bfc-1d05-11eb-2f4e-331c0302a14b
# ╟─1fb01056-1d06-11eb-241f-49b9b90929dd
# ╟─82b3677a-1d06-11eb-1a41-c3a9b29e9995
# ╟─834f9d50-60c9-11eb-3207-3bb56de82801
# ╟─aede18f0-360f-11eb-182f-0fbb1762f733
# ╟─21e84c2a-5407-11eb-3bbf-6590b0ca21eb
# ╟─19b432e6-1d14-11eb-1770-fda8ed6ad743
