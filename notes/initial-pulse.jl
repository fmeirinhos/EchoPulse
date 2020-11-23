### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 54c8d9b4-2d99-11eb-242b-49681ab38480
begin
	using KadanoffBaym

	import SpecialFunctions: erf, erfc
	import LinearAlgebra: diag

	using Plots, LaTeXStrings
	# default(fmt = :png)
end

# ╔═╡ a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
begin
	# Gaussian pulse parameters
	const tc = 15.  # peak arrival time
	const Ω = 0.3 # frequency bandwidth of the pulse

	function ξ(t)
		return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
	end

	# Pump and loss terms
	const Γ = 5.0
	λ(t::Float64) = iszero(Γ) ? 0.0 : Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2)))
	γ(t::Float64) = iszero(Γ) ? 0.0 : Γ * ξ(t)

	# Photon field Hamiltonian
	h(t::Float64) = 5.0 - 0.5im * (λ(t) + γ(t))

	# Artificial width of the peak
	Σ = 1.0

	# Initial field occupation
	n0 = 0.0

	# Initial conditions
	GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), Lesser)
	GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), Greater)

	# Pack them
	u0 = [GᵞL, GᵞG];

	# Define commutator
	⋊(a,b) = a * b - b * adjoint(a)

	function f_vert(u, times, t1, t2)
		t = times[t1]
		GL, GG = u[1], u[2]

		rhsL = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GL[t1,t2])
		rhsG = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GG[t1,t2])
		return [rhsL, rhsG]
	end

	function f_diag(u, times, t1)
		t = times[t1]
		println("t: $t")
		GL, GG = u[1], u[2]

		rhsL = -1.0im * (h(t) ⋊ GL[t1,t1] + 1.0im * γ(t) * (GL[t1,t1] + GG[t1,t1]))
		rhsG = -1.0im * (h(t) ⋊ GG[t1,t1] + 1.0im * λ(t) * (GG[t1,t1] - GL[t1,t1]) + 2.0im * γ(t) * GG[t1,t1])
		return [rhsL, rhsG]
	end
end

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
begin
	tspan = (0.0,30.0)
	sol, _ = kbsolve(f_vert, f_diag, u0, tspan; 
		atol=1e-6, rtol=1e-3)
end;

# ╔═╡ 26b0aed4-2d9a-11eb-127b-cbc0eabee1be
begin
	using Interpolations

	itp = interpolate((sol.t,sol.t), GᵞL.data, Gridded(Linear()))
	time_eq = range(sol.t[1], stop=sol.t[end], length=length(sol.t))

	ρ, (ωs, ts) = wigner_transform([itp(t1,t2) for t1=time_eq, t2=time_eq], ts=time_eq);

	heatmap(ts, ωs, real(ρ); xlabel=L"$T$", ylabel=L"$\omega$", ylims=(-10,10))
end

# ╔═╡ 00b4636a-2d9a-11eb-2e36-19d6f90eba2a
begin
	plot(sol.t, λ.(sol.t), label="Loss term")
	plot!(sol.t, γ.(sol.t), label="Pump term")
	plot!(twinx(), sol.t, ξ.(sol.t), color=:green, legend=:right, label="Pulse", line=:dot, xlabel=L"t")
end

# ╔═╡ 0c6a841e-2d9a-11eb-1d8c-151d18116e55
begin
	plot(sol.t, GᵞL.data |> diag .|> imag .|> -, label="Lesser")
	plot!(sol.t, GᵞG.data |> diag .|> imag .|> -, label="Greater")
	plot!(twinx(), sol.t, ξ.(sol.t), color=:green, legend=:right, label="Pulse", line=:dot, xlabel=L"t")
end

# ╔═╡ 53737820-2d9a-11eb-2da7-c553a8a2c56e
# Possible analytic pulse
f(t,t′) = -1.0im * exp(-1.0im * (t-t′) * (5.0 - 1.0im * sign(t-t′) * Σ)) * ξ(0.5 * (t+t′))

# ╔═╡ 56f857e0-2d9a-11eb-38a9-bf4566aabc1c
heatmap([f(a,b) |> imag for a in sol.t, b in sol.t])

# ╔═╡ 5bdcd6c8-2d9a-11eb-22a7-45a53df95187
heatmap(GᵞL.data |> imag)

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═00b4636a-2d9a-11eb-2e36-19d6f90eba2a
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
# ╠═26b0aed4-2d9a-11eb-127b-cbc0eabee1be
# ╠═53737820-2d9a-11eb-2da7-c553a8a2c56e
# ╠═56f857e0-2d9a-11eb-38a9-bf4566aabc1c
# ╠═5bdcd6c8-2d9a-11eb-22a7-45a53df95187
