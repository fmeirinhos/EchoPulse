### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6a62bdd8-7139-11eb-157e-8b433dd0177b
using DSP.FFTW, Interpolations, NCA, KadanoffBaym, Plots

# ╔═╡ 3b1f4142-8f9b-11eb-37ac-8b3898176679
md"""
This notebook is for testing how many k-states for the c-electrons should be taken to approximately match the continuum c-electron bath in equilibrium ``\sum_k G^c_k(\omega)``.

It's important to note that as `` t \gg 1`` some spurious features may arise which are probably not desirable. 
"""

# ╔═╡ ba3e2e22-713a-11eb-108c-edd5ce13de5b
function ft(xs, ys; mode=+1)
  # @assert ordered_xs
  dx = xs[2] - xs[1]

  x̂s = fftfreq(length(xs), -2pi/dx)
  is = sortperm(x̂s)

  ℯⁱᵠ = dx * exp.(mode * 1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

  return x̂s[is], (ℯⁱᵠ .* (mode == 1 ? fft(ys) : ifft(ys)))[is]
end

# ╔═╡ 8ba39052-713a-11eb-2993-8f2045a516eb
begin
	ω = range(-20.0, 20.0, length=2^13)
	β = 10
	
	ρ(ω) = NCA.gaussian(ω, 0.0, 0.5)
	f(ω) = NCA.nf(ω, β)
	
	ts, G_ = ft(ω, im * ρ.(ω) .* f.(-ω), mode=-1)
	GG_exact = extrapolate(interpolate((ts,), -G_, Gridded(Linear())), 0)
end

# ╔═╡ a8a20078-7525-11eb-37c1-073d70915987
begin
	using AMR
	
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
	
	(x, y) = sample_function(ρ, range(-2.0, 2., length=2^4); tol=1e-6)
	
	GG_amr(t,t′) = integrate(x, -1.0im * exp.(-1.0im * (t-t′) .* x) .* y .* f.(-x))
end

# ╔═╡ 9ede3c9a-7139-11eb-2db7-4bd9267ba6a0
begin
	ω_local = range(-2.0, 2.0, length=2^9)
		
	GG_sum(t,t′) = (ω_local[2] - ω_local[1]) * sum(-1.0im * exp(-1.0im * (t-t′) * ω) * ρ(ω) * f(-ω) for ω in ω_local)
end

# ╔═╡ 5da544ac-713a-11eb-29b2-59d2d3d33146
begin
	t = range(0.0, 700., length=10000)
	plot(t, GG_exact.(t) |> real)
	plot!(t, GG_sum.(t, 0.0) |> real)
	# plot!(t, GG_amr.(t, 0.0) |> real)
end

# ╔═╡ e450f60e-7527-11eb-30e6-e39f2c497a78
plot(t, GG_exact.(t) .- GG_sum.(t, 0.0) |> real)

# ╔═╡ Cell order:
# ╠═3b1f4142-8f9b-11eb-37ac-8b3898176679
# ╠═6a62bdd8-7139-11eb-157e-8b433dd0177b
# ╠═ba3e2e22-713a-11eb-108c-edd5ce13de5b
# ╠═8ba39052-713a-11eb-2993-8f2045a516eb
# ╠═9ede3c9a-7139-11eb-2db7-4bd9267ba6a0
# ╠═a8a20078-7525-11eb-37c1-073d70915987
# ╠═5da544ac-713a-11eb-29b2-59d2d3d33146
# ╠═e450f60e-7527-11eb-30e6-e39f2c497a78
