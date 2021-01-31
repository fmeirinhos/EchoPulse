### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 53b29afc-6321-11eb-33cb-013865fbe485
begin
	using DSP, NCA, NLsolve
	using FFTW, Interpolations, KadanoffBaym
	using Parameters: @unpack
	
	using LinearAlgebra: diag
	using Plots, LaTeXStrings
	Plots.gr()
end;

# ╔═╡ 48fe615c-6323-11eb-2c99-29b2dba8c971
# This controls all the parameters of our model
begin
	# temperature
	β = 1.0e0  
	
	#= SIAM parameters =#
	V₀=sqrt(0.08)	# hybridization strength
	ε₀f=-0.5		# d-electron energy
	σ=0.5			# gaussian cdos width

	#= quantum light parameters =#
	Γ₀=sqrt(0.01) 	# light-matter coupling
	ω₀=1e0 			# photon energy
	Ω=4e-1			# frequency bandwitdh of the gaussian pulse
	tc=8.0e1 		# peak arrival time
	Π=5.0e1 		# maximum photon occupation
	
	# Pulse shape s.t. ∫dt ξ(t)^2 = 1
	ξ(t) = (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
	
	# Integration bounds
	time_span = (0.0, 200.0)
end;

# ╔═╡ b007ee22-6323-11eb-11cf-c1e43c3aac57
# Solve the problem in equilibrium
imG_t, G = begin

	# Equilibrium model
	nca = FreqNCA(
		model=SIAM(
			V₀=V₀,
			ε₀f=ε₀f,
			tilde=true,
			Nc₀=x-> 2pi * NCA.gaussian(x, -0.0, σ)),
		β=β,
		mesh=LinearMesh(-20., +20., 2^16))

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
end;

# ╔═╡ 91f10e24-63cc-11eb-24ac-f32af38df214
begin
	cecu6_bandwidth = 2.0# eV
	cdos_bandwidth = 6σ # 3σ
	to_eV = cecu6_bandwidth / cdos_bandwidth
	fwmh_gaussian = 2 * sqrt(2 * log(2))
	fwmh_lorentzi = 2
	ω_to_eV(ω) = round(ω * to_eV , digits=3)
	ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2, digits=3)
	ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2), digits=4)
	ω_to_K(ω) = round(ω * to_eV * 1.1604518e4, digits=3)
	
	# Kondo frequency
	ωK = NCA.kondo_temp(nca.mesh, nca.data.Nc ./ (2pi), ε₀f, V₀)
	
md"
#### Summary
Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $cecu6_bandwidth ``eV``, the chosen SIAM parameters were
1. ``\varepsilon_d =`` $(ω_to_eV(ε₀f)) ``eV`` (``d-``orbital energy)
2. ``V_0 =  `` $(ω_to_eV(V₀)) ``eV`` (hybridisation strength)
3. ``T_K = `` $(ω_to_K(ωK)) ``K`` (Kondo temperature: $(ω_to_ps(ωK)) ``ps``)
4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
and the gaussian-shaped pulse parameters were
5. ``\omega = `` $(ω_to_THz(ω₀)) ``THz`` (pulse energy)
6. ``\Omega = `` $(ω_to_THz(Ω)) ``THz``  (pulse bandwidth: $(6ω_to_ps(Ω)) ``ps``)
7. ``\Gamma = `` $(ω_to_eV(Γ₀)) ``eV`` (light-matter coupling)
8. ``\Pi = `` $(round(Π, digits=3)) ``n_\gamma`` (maximum photon-field occupation)

The system will be integrated from $(ω_to_ps(1/time_span[1])) to $(ω_to_ps(1/time_span[2])) ps with the light-pulse reaching its maximum at $(ω_to_ps(1/tc)) ps.

###### Questions
1. Do we need to shift the photons energy by the chemical potential?
2. How to convert maximum photon occupation to ``MV cm^{-1}``
3. Fix angular frequency problems: ``2\pi \omega t``
"
end

# ╔═╡ 88833404-63cf-11eb-1b09-914db36d396e
begin
	imGd = NCA.xcorr(nca.mesh, nca.data.fermi .* imG_t[:,1], nca.data.fermi⁻ .* imG_t[:,2]) + NCA.xcorr(nca.mesh, nca.data.fermi⁻ .* imG_t[:,1], nca.data.fermi .* imG_t[:,2])
	
	plot(ω_to_eV.(nca.mesh.mesh), nca.data.Nc;
		label=L"$A_c$", 
		linecolor=:green,
		legend=:topleft,
		xlabel=L"\omega \quad (eV)",#L"$\omega/D$",
		xlims=ω_to_eV.((-5.0, 5.0)),
		title=L"\textrm{Equilibrium spectral functions}"
		)
	
	plot!(ω_to_eV.(nca.mesh.mesh), imGd,
		label=L"$A_d$",
		linecolor=:darkorange)
	
	vline!(ω_to_eV.([ε₀f]);
		label=L"\varepsilon_d",
		line=(:dot, 1.0, :blue, 0.5))
	
	vspan!(ω_to_eV.([-ωK/2, ωK/2]);
		fillcolor=:grey,
		alpha=0.3,
		label=L"\omega_K")
end

# ╔═╡ 4d9ffbbc-63e6-11eb-2ff0-d7b1b0ff15f7
begin
	function ft(xs, ys; mode=+1)
		# @assert ordered_xs
		dx = xs[2] - xs[1]

		x̂s = FFTW.fftfreq(length(xs), -2pi/dx)
		is = sortperm(x̂s)

		ℯⁱᵠ = dx * exp.(mode * 1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

		return x̂s[is], (ℯⁱᵠ .* (mode == 1 ? fft(ys) : ifft(ys)))[is]
	end
	
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
	
	function dyson_lesser(ts, G0L, G0G, TL, TG)
		GL = diag(G0L)

		for t in eachindex(ts)
			@views GL[t] += integrate(ts[1:t], [G0G[t,t′] * (
						+ integrate(ts[1:t′], TG[t′,1:t′] .* G0L[1:t′,t])
						+ integrate(ts[t′:t], TL[t′,t′:t] .* G0L[t′:t,t])
						- integrate(ts[1:t], TL[t′,1:t] .* G0G[1:t,t])
							) for t′ in 1:t])

			@views GL[t] -= integrate(ts[1:t], [G0L[t,t′] * (
						+ integrate(ts[1:t], TG[t′,1:t] .* G0L[1:t,t])
						- integrate(ts[t′:t], TG[t′,t′:t] .* G0G[t′:t,t])
						- integrate(ts[1:t′], TL[t′,1:t′] .* G0G[1:t′,t])
							) for t′ in 1:t])
		end
		return GL
	end
end;

# ╔═╡ 07b90370-63b7-11eb-2de1-97e4910b827d
# Initial condition for the time evolution
begin
	# Update gauge chem potential
	λ₀ = nca.model.λ
	
	# frequency integration
	∫dω(f) = integrate(nca.mesh.mesh, f) / (2π)
	
	ts, L_ = ft(nca.mesh, 1.0im/(2pi) * nca.data.Nc .* nca.data.fermi, mode=-1)
	GᶜL = extrapolate(interpolate((ts,), +L_, Gridded(Linear())), 0)
	ts, G_ = ft(nca.mesh, 1.0im/(2pi) * nca.data.Nc .* nca.data.fermi⁻, mode=-1)
	GᶜG = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)
	
	GᶠL = GreenFunction(+1.0im * ∫dω(nca.data.fermi .* 2imG_t[:,2]) * ones(1,1), Lesser)
	GᶠG = GreenFunction(-1.0im * ones(1,1), Greater)
	ΣᶠL = GreenFunction(zero(GᶠL.data), Lesser)
	ΣᶠG = GreenFunction(zero(GᶠG.data), Greater)
	
	GᵇL = GreenFunction(-1.0im * ∫dω(nca.data.fermi .* 2imG_t[:,1]) * ones(1,1), Lesser) 
	GᵇG = GreenFunction(-1.0im * ones(1,1), Greater)
	ΣᵇL = GreenFunction(zero(GᵇL.data), Lesser)
	ΣᵇG = GreenFunction(zero(GᵇG.data), Greater)
	
	n_ξ(t) = (Π / Ω * sqrt(2pi)) * ξ(t)^2
	GᵞL₀(t,t′) = -1.0im * exp(-1.0im * (t-t′) * ω₀) * n_ξ(0.5(t+t′))
	GᵞG₀(t,t′) = -1.0im * exp(-1.0im * (t-t′) * ω₀) * (n_ξ(0.5(t+t′)) + 1)
	TᵞL = GreenFunction(zero(GᵇL.data), Lesser)
	TᵞG = GreenFunction(zero(GᵇG.data), Greater)
end;

# ╔═╡ a9b1c1ee-6321-11eb-0308-a7836c81d5ba
# Kadanoff-Baym equations
begin
	function f_vert_time(_,times,t1,t2)
		
		function ∫dt(i,j,A,B)
			retval = zero(eltype(A))
			@fastmath @simd for k in min(i,j):(max(i,j)-1)
				@inbounds retval += (times[k+1] - times[k]) * (
					A[t1,k]*B[k,t2] + A[t1,k+1]*B[k+1,t2])
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
	end
	
	function f_diag_time(_,times,t1)
		
		function ∫dt(A,B)
			retval = zero(eltype(A))
			@fastmath @simd for k in 1:(t1-1)
				@inbounds retval += (times[k+1] - times[k]) * (
					A[t1,k]*B[k,t1] + A[t1,k+1]*B[k+1,t1])
			end
			return 1//2 * retval
		end
		
		# pseudo fermion
		dGᶠL = -1.0im * (∫dt(ΣᶠG,GᶠL) - ∫dt(ΣᶠL,GᶠG) + ∫dt(GᶠL,ΣᶠG) - ∫dt(GᶠG,ΣᶠL))
		dGᶠG = 0.0

		# slave boson
		dGᵇL = -1.0im * (∫dt(ΣᵇG,GᵇL) - ∫dt(ΣᵇL,GᵇG) + ∫dt(GᵇL,ΣᵇG) - ∫dt(GᵇG,ΣᵇL))
		dGᵇG = 0.0

		return [dGᶠL, dGᶠG, dGᵇL, dGᵇG]
	end
	
	function calculate_Σ_time(times,t1,t2)
		if (n = size(GᶠL,2)) > size(ΣᶠL,2)
			resize!(ΣᶠL,n); resize!(ΣᶠG,n)
			resize!(ΣᵇL,n); resize!(ΣᵇG,n)
			resize!(TᵞL,n); resize!(TᵞG,n)
		end

		V²L = V₀^2 + Γ₀^2 * 1.0im * (GᵞL₀(times[t1],times[t2]) + GᵞG₀(times[t2],times[t1]))
		V²G = V₀^2 + Γ₀^2 * 1.0im * (GᵞG₀(times[t1],times[t2]) + GᵞL₀(times[t2],times[t1]))

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
end;

# ╔═╡ f00fd49a-63d6-11eb-3a91-710323a7bd40
begin
	@time sol, caches = kbsolve(f_vert_time, f_diag_time,
		[GᶠL,GᶠG,GᵇL,GᵇG],
		time_span;
		atol=1e-6, rtol=1e-3, dtini=1e-6, dtmax=10.,
		update_time =(t,t1,t2)->calculate_Σ_time(t,t1,t2))

	times = sol.t
end;

# ╔═╡ e0109d4c-63d9-11eb-07ee-1f15ec786ad7
begin
	# Spectral function
	ρt = +1.0im * (GᶠG.data .* conj(GᵇL.data) - GᶠL.data .* conj(GᵇG.data))
	ρτ, (τs, Ts) = wigner_transform_itp(ρt, times; fourier=false)
	ρω, (ωs, Ts) = wigner_transform_itp(ρt, times; fourier=true)
	
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
	G0L = [GᵞL₀(t,t′) for t in times, t′ in times]
	G0G = [GᵞG₀(t,t′) for t in times, t′ in times]
	
	TᵞL.data = TᵞL.data[1:length(times),1:length(times)]
	TᵞG.data = TᵞG.data[1:length(times),1:length(times)]
	
	GL = dyson_lesser(times, G0L, G0G, TᵞL, TᵞG)
	
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

# ╔═╡ Cell order:
# ╟─53b29afc-6321-11eb-33cb-013865fbe485
# ╠═48fe615c-6323-11eb-2c99-29b2dba8c971
# ╟─91f10e24-63cc-11eb-24ac-f32af38df214
# ╟─88833404-63cf-11eb-1b09-914db36d396e
# ╟─e0109d4c-63d9-11eb-07ee-1f15ec786ad7
# ╟─b007ee22-6323-11eb-11cf-c1e43c3aac57
# ╟─07b90370-63b7-11eb-2de1-97e4910b827d
# ╟─a9b1c1ee-6321-11eb-0308-a7836c81d5ba
# ╟─f00fd49a-63d6-11eb-3a91-710323a7bd40
# ╟─4d9ffbbc-63e6-11eb-2ff0-d7b1b0ff15f7
