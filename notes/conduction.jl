### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6a62bdd8-7139-11eb-157e-8b433dd0177b
using DSP.FFTW, Interpolations, NCA, KadanoffBaym, Plots

# ╔═╡ ba3e2e22-713a-11eb-108c-edd5ce13de5b
function ft(xs, ys; mode=+1)
  # @assert ordered_xs
  dx = xs[2] - xs[1]

  x̂s = fftfreq(length(xs), -2pi/dx)
  is = sortperm(x̂s)

  ℯⁱᵠ = dx * exp.(mode * 1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

  return x̂s[is], (ℯⁱᵠ .* (mode == 1 ? fft(ys) : ifft(ys)))[is]
end

# ╔═╡ 7a82c94c-7139-11eb-0009-e5f79179cd76
begin
	ω = range(-10.0, 10.0, length=2^10)
	ρ = NCA.gaussian.(ω, 0.0, 0.5)
	β = 10
	f(ω) = NCA.nf(ω, β)
end

# ╔═╡ 9ede3c9a-7139-11eb-2db7-4bd9267ba6a0
begin
	GL(t,t′) = (ω[2] - ω[1]) * sum(1.0im * exp(-1.0im * (t-t′) * ω[i]) * ρ[i] * f(ω[i]) for i in eachindex(ω))
	GG(t,t′) = (ω[2] - ω[1]) * sum(-1.0im * exp(-1.0im * (t-t′) * ω[i]) * ρ[i] * f(-ω[i]) for i in eachindex(ω))
end

# ╔═╡ 8ba39052-713a-11eb-2993-8f2045a516eb
begin
	ts, L_ = ft(ω, im * ρ .* f.(ω), mode=-1)
	GL_ = extrapolate(interpolate((ts,), +L_, Gridded(Linear())), 0)
	ts, G_ = ft(ω, im * ρ .* f.(-ω), mode=-1)
	GG_ = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)
end

# ╔═╡ 5da544ac-713a-11eb-29b2-59d2d3d33146
begin
	t = range(0.0, 10., length=100)
	plot(t, GG.(t, 0.0) |> real)
	plot!(t, GG_.(t) |> real)
end

# ╔═╡ Cell order:
# ╠═6a62bdd8-7139-11eb-157e-8b433dd0177b
# ╠═ba3e2e22-713a-11eb-108c-edd5ce13de5b
# ╠═7a82c94c-7139-11eb-0009-e5f79179cd76
# ╠═9ede3c9a-7139-11eb-2db7-4bd9267ba6a0
# ╠═8ba39052-713a-11eb-2993-8f2045a516eb
# ╠═5da544ac-713a-11eb-29b2-59d2d3d33146
