### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ ed50b1fe-6a1e-11eb-08ab-95d436eef99c
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end;

# ╔═╡ 292f7a80-6a1e-11eb-144e-b96c3e89f430
begin
	X = ingredients("../src/EchoPulse.jl")
	import .X.EchoPulse: PhotonAssistedModel, FrequencyData, TimeData, f_vt, f_dt, Σ_t
	
	using KadanoffBaym
	using LinearAlgebra: diag
	using UnPack
	using Plots, LaTeXStrings
end;

# ╔═╡ 49afc5ac-6a24-11eb-3446-715b1b8c2eb7
model, tmax = begin 
	β  = 5.0e2  # temperature

	model = PhotonAssistedModel(
		#= SIAM parameters =#
		V₀=sqrt(0.03),	# hybridization strength
		ε₀f=-0.3,		# d-electron energy

		#= quantum light parameters =#
		Γ₀=sqrt(0.005), # light-matter coupling
		ω₀=5e-2, 		# photon energy
		Ω=3e-2,			# frequency bandwitdh of the gaussian pulse
		tc=1.5e2,		# peak arrival time
		Π=1.0e1 		# maximum photon occupation
	)
	
	model, 300.0
end;

# ╔═╡ 89141fc4-6a2c-11eb-11d4-432a71ae7968
begin
	ω_data = FrequencyData(model, range(-3,3,length=2^14), β);
	t_data, t0 = TimeData(model, ω_data; 
		# timematrix=true, 
		# ωᶜ=collect(range(-4*model.σ,4*model.σ,length=2^8))
	)
	
	@unpack GᶜL, GᶜG, GᶠL, GᶠG, GᵇL, GᵇG, GᵞL₀, GᵞG₀ = t_data
end;

# ╔═╡ 0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
begin
	@time sol = kbsolve(
		(_,times,t,t′) -> f_vt(model, t_data, times, t, t′),
		(_,times,t) -> (println(times[t]); f_dt(model, t_data, times, t)),
		isnothing(t_data.ωᶜ) ? [GᶠL,GᶠG,GᵇL,GᵇG] : [[GᶠL,GᶠG,GᵇL,GᵇG];GᶜL;GᶜG],
		(t0, tmax);
		atol=1e-6, rtol=1e-3,
		update_time =(times,t,t′)->Σ_t(model, t_data, times, t, t′))

	times = sol.t
end;

# ╔═╡ fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
begin
	cecu6_half_bandwidth = 2.0
	cdos_half_bandwidth = 1
	to_eV = cecu6_half_bandwidth / cdos_half_bandwidth
	ω_to_eV(ω) = round(ω * to_eV , digits=3)
	ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2, digits=3)
	ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2), digits=4)
	ω_to_K(ω) = round(ω * to_eV * 1.1604518e4, digits=3)
md"
#### Summary
Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the chosen SIAM parameters were
1. ``\varepsilon_d =`` $(ω_to_eV(model.ε₀f)) ``eV`` (``d-``orbital energy)
2. ``V_0 =  `` $(ω_to_eV(model.V₀)) ``eV`` (hybridisation strength)
3. ``T_K = `` $(ω_to_K(ω_data.ωK)) ``K`` (Kondo temperature: $(ω_to_ps(ω_data.ωK)) ``ps``)
4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
and the gaussian-shaped pulse parameters were
5. ``\omega = `` $(ω_to_THz(model.ω₀)) ``THz`` (pulse energy)
6. ``\Omega = `` $(ω_to_THz(model.Ω)) ``THz``  (pulse bandwidth: $(6ω_to_ps(model.Ω)) ``ps``)
7. ``\Gamma = `` $(ω_to_eV(model.Γ₀)) ``eV`` (light-matter coupling)
8. ``\Pi = `` $(round(model.Π, digits=3)) ``n_\gamma`` (maximum photon-field occupation)

The system will be integrated for to $(ω_to_ps(1/tmax)) ps with the light-pulse reaching its maximum at $(ω_to_ps(1/model.tc)) ps.

###### Questions
1. Do we need to shift the photons energy by the chemical potential?
2. How to convert maximum photon occupation to ``MV cm^{-1}``
"
end

# ╔═╡ 56d36398-6a32-11eb-3f78-01155850c5d2
begin	
	plot(ω_data.nca.mesh .|> ω_to_eV, ω_data.nca.data.ρc;
		label=L"$A_c$", 
		linecolor=:green,
		legend=:topleft,
		xlabel=L"\omega \quad (eV)",#L"$\omega/D$",
		xlims=extrema(ω_data.nca.mesh) .|> ω_to_eV,
		title=L"\textrm{Equilibrium spectral functions}"
		)
	
	plot!(ω_data.nca.mesh .|> ω_to_eV, 2ω_data.G_d |> imag,
		label=L"$A_d$",
		linecolor=:darkorange)
	
	vline!([model.ε₀f] .|> ω_to_eV;
		label=L"\varepsilon_d",
		line=(:dot, 1.0, :blue, 0.7))
	
	vspan!([-ω_data.ωK/2, ω_data.ωK/2] .|> ω_to_eV;
		fillcolor=:grey,
		alpha=0.3,
		label=L"\omega_K")
end

# ╔═╡ eda788ba-6f84-11eb-24ee-17e261d30bc1
begin
	# Spectral function
	ρt = GᶠL.data .* conj(-GᵇG.data) - GᶠG.data .* conj(-GᵇL.data)
	
	ρτ, (τs, Ts) = wigner_transform_itp(ρt, times; fourier=false)
	ρω, (ωs, Ts) = wigner_transform_itp(ρt, times; fourier=true)
	
	h1 = heatmap(ω_to_ps.(inv.(Ts)), ωs, ρω |> imag;
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
		clims=extrema(ρω |> imag))
	
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
	G0L = [GᵞL₀(1//2 * (t+t′), t-t′) for t in times, t′ in times]
	G0G = [GᵞG₀(1//2 * (t+t′), t-t′) for t in times, t′ in times]
	
# 	TᵞL.data = TᵞL.data[1:length(times),1:length(times)]
# 	TᵞG.data = TᵞG.data[1:length(times),1:length(times)]
	
# 	GL = dyson_lesser(times, G0L, G0G, TᵞL, TᵞG)
	
	p1 = plot!(plt, ω_to_ps.(inv.(times)), G0L |> (-) ∘ imag ∘ diag;
		label=nothing,
		line=(:green, 3.0, 0.2),
		legend=:topright,
		xlabel=L"T \quad (ps)",#L"T\;D",
		xlims=ω_to_ps.(inv.(times)),
		grid=false)
	# p1 = plot!(plt, ω_to_ps.(inv.(times)), GL |> (-) ∘ imag;
	# 	label=L"n_\gamma",
	# 	line=(:green))
	
	# Kondo time-scale
	t_max_pulse = times[findmax(G0L |> (-) ∘ imag ∘ diag)[2]]
	vspan!(plt, ω_to_ps.(inv.(t_max_pulse .+ 1/ω_data.ωK .* [-0.5, +0.5]));
		fillcolor=:grey,
		alpha=0.3,
		label=L"T_K")
	
	plot(h1, p1, colorbar, layout = @layout [grid(2, 1) a{0.05w}])
	# savefig("ad_wigner.pdf")
end

# ╔═╡ Cell order:
# ╟─ed50b1fe-6a1e-11eb-08ab-95d436eef99c
# ╟─292f7a80-6a1e-11eb-144e-b96c3e89f430
# ╠═49afc5ac-6a24-11eb-3446-715b1b8c2eb7
# ╟─89141fc4-6a2c-11eb-11d4-432a71ae7968
# ╟─0cdaa3dc-6f7d-11eb-145d-2790c94fec9d
# ╟─fc8cca44-6a1f-11eb-28ac-a5f3445d4f56
# ╟─56d36398-6a32-11eb-3f78-01155850c5d2
# ╟─eda788ba-6f84-11eb-24ee-17e261d30bc1
