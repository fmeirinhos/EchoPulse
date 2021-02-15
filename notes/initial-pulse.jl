### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 54c8d9b4-2d99-11eb-242b-49681ab38480
begin
	using Interpolations, DSP.FFTW, KadanoffBaym
	import LinearAlgebra: diag

	using Plots, LaTeXStrings
	# default(fmt = :png)
end

# ╔═╡ a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
begin
	# Gaussian pulse parameters
	const tc = 100.  # peak arrival time
	const Ω = 0.1 	# frequency bandwidth of the pulse

	function ξ(t)
		return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
	end

	# Photon field Hamiltonian
	ω₀ = 1.0

	# Initial field occupation
	n0 = ξ(0.0)^2 / Ω
	@assert !iszero(n0)

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
		
		rhsL = -1.0im * ω₀ * GL[t1,t2] - Ω^2 / 2 * (times[t1] - tc) * GL[t1,t2] 
		rhsG = -1.0im * ω₀ * GG[t1,t2] - Ω^2 / 2 * (times[t1] - tc) * GL[t1,t2] 
		return [rhsL, rhsG]
	end

	function f_diag(u, times, t1)
		t = times[t1]
		println("t: $t")
		GL, GG = u[1], u[2]
		
		rhsL = -1.0 * (Ω^2 * (times[t1] - tc)) * GL[t1,t1]
		rhsG = -1.0 * (Ω^2 * (times[t1] - tc)) * GL[t1,t1]
		
		return [rhsL, rhsG]
	end
end

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
begin
	tspan = (0.0,200.0)
	sol = kbsolve(f_vert, f_diag, u0, tspan; 
		atol=1e-6, rtol=1e-3)
end;

# ╔═╡ 0c6a841e-2d9a-11eb-1d8c-151d18116e55
begin
	plot(sol.t, GᵞL.data |> diag .|> imag .|> -, label="Lesser")
	plot!(sol.t, GᵞG.data |> diag .|> imag .|> -, label="Greater")
	plot!(twinx(), sol.t, ξ.(sol.t), color=:green, legend=:right, label="Pulse", line=:dot, xlabel=L"t")
end

# ╔═╡ 26b0aed4-2d9a-11eb-127b-cbc0eabee1be
begin
	ρ, (ωs, ts) = wigner_transform_itp(GᵞL.data, sol.t);

	heatmap(ts, ωs, real(ρ); xlabel=L"$T$", ylabel=L"$\omega$")
end

# ╔═╡ 53737820-2d9a-11eb-2da7-c553a8a2c56e
# Possible analytic pulse
f(t,t′) = -1.0im * exp(-1.0im * (t-t′) * ω₀) * ξ(0.5 * (t+t′))^2 * exp(-(t-t′)^2 * (Ω^2/4))

# ╔═╡ 56f857e0-2d9a-11eb-38a9-bf4566aabc1c
heatmap(sol.t, sol.t, [f(a,b) |> imag for a in sol.t, b in sol.t])

# ╔═╡ 5bdcd6c8-2d9a-11eb-22a7-45a53df95187
heatmap(sol.t, sol.t, GᵞL.data |> imag)

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
# ╠═26b0aed4-2d9a-11eb-127b-cbc0eabee1be
# ╠═53737820-2d9a-11eb-2da7-c553a8a2c56e
# ╠═56f857e0-2d9a-11eb-38a9-bf4566aabc1c
# ╠═5bdcd6c8-2d9a-11eb-22a7-45a53df95187
