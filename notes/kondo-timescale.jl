### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 0b3c1108-2509-11eb-28c2-bb575678c04a
begin
	using NCA
	using KadanoffBaym
	using KadanoffBaym.FFTW: fftfreq, fft
	using LinearAlgebra
	using Plots
end

# ╔═╡ 20dfa5c6-2509-11eb-049a-37a4b96cea75
nca = FreqNCA(
	model=NCA.SIAM(
      V₀=sqrt(0.1),
      ε₀f=-0.3,
      tilde=true),
	β=1e3,
    mesh=LinearMesh(-5.0, 5.0, 2^12)
)

# ╔═╡ 4f90dc0a-2509-11eb-305a-e7fcb0ba1e50
x = NCA.slave_particle_imG(nca, NCA.solve!(nca; project=false)) ./ pi

# ╔═╡ a5ef4166-2595-11eb-0a12-dfc7e0794078
begin
	ωs = nca.mesh.mesh
	dω= ωs[2] - ωs[1]
	
	ts = fftfreq(length(ωs), 2π/dω)
	is = sortperm(ts)
	
	ℯⁱᵠ = dω * exp.(1.0im * ωs[1] .* ts)
	
	L = mapslices(x -> ℯⁱᵠ .* fft(x .* nca.data.fermi), x, dims=1)  # Lesser
	G = mapslices(x -> ℯⁱᵠ .* fft(x .* nca.data.fermi⁻), x, dims=1) # Spectral
end

# ╔═╡ 5144c37a-2509-11eb-09c6-a5b11391865c
begin
	plot(ωs, x[:,1])
	plot!(ωs, x[:,2])
end

# ╔═╡ d5927c1a-25b2-11eb-1f8a-2f5e572722e7
begin
	X = repeat(x[:,1], outer=[1,length(ωs)])
	
	heatmap(
		ts[is][1:10:end] .+ ts[is[end]], 
		ωs[1:10:end], 
		X[1:10:end,1:10:end] |> real, xlabel="t", ylabel="ω")
end

# ╔═╡ a34a132c-26b0-11eb-2270-b95c8425be03
begin	
	plot(ts[is], L[is,1] .|> real, label="GL_b")
	plot!(ts[is], L[is,2] .|> real, label="GL_f")
	xlims!(-100,100)
end

# ╔═╡ df7b1ab6-25b2-11eb-3d39-69d22eee1ccc
begin	
	plot(ts[is], G[is,1] .|> real, label="GG_b")
	plot!(ts[is], G[is,2] .|> real, label="GG_f")
	xlims!(-100,100)
end

# ╔═╡ e76a6498-25b2-11eb-1172-1773740ccf31
begin
	Y = repeat(L[:,2], outer=[1,length(ωs)])
	
	heatmap(
		ts[is][1:10:end] .+ ts[is[end]],
		ts[is][1:10:end],
		Y[is,:][1:10:end,1:10:end] |> real)
end

# ╔═╡ edae77b8-25b2-11eb-3859-f338370be8ba
begin
	Z = begin
		Z_ = zero(Y)
		
		L2 = size(Z_,1) ÷ 2
		for i in 1:L2
			view(Z_, diagind(Z_, i)) .= L[i]
			view(Z_, diagind(Z_, -i)) .= L[2L2-i]
		end
		Z_
	end
	
	heatmap(
		ts[is][1:10:end] .+ ts[is[end]],
		ts[is][1:10:end],
		Z[1:10:end,1:10:end] |> real)
end

# ╔═╡ Cell order:
# ╠═0b3c1108-2509-11eb-28c2-bb575678c04a
# ╠═20dfa5c6-2509-11eb-049a-37a4b96cea75
# ╠═4f90dc0a-2509-11eb-305a-e7fcb0ba1e50
# ╠═a5ef4166-2595-11eb-0a12-dfc7e0794078
# ╠═5144c37a-2509-11eb-09c6-a5b11391865c
# ╠═d5927c1a-25b2-11eb-1f8a-2f5e572722e7
# ╠═a34a132c-26b0-11eb-2270-b95c8425be03
# ╠═df7b1ab6-25b2-11eb-3d39-69d22eee1ccc
# ╠═e76a6498-25b2-11eb-1172-1773740ccf31
# ╠═edae77b8-25b2-11eb-3859-f338370be8ba
