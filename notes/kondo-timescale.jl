### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 0b3c1108-2509-11eb-28c2-bb575678c04a
begin
	include("../src/EchoPulse.jl")
	using DSP, NCA, NLsolve
	using LinearAlgebra
	using Plots
end

# ╔═╡ 20dfa5c6-2509-11eb-049a-37a4b96cea75
nca = let
	β = 1e3
    siam = SIAM(V₀=sqrt(0.08), ε₀f=-0.3)
    mesh = LinearMesh(-4.0, +4.0, 2^12)
    data = SIAMData(siam, mesh, β, ω -> 2pi * NCA.gaussian(ω, 0.0, 1/3))

    FreqNCA(model=siam, mesh=mesh, data=data, β=β)
end;

# ╔═╡ a777bba2-8f9c-11eb-0d93-870827d56d7b
imG_t = let
    x0 = abs2(nca.model.V₀) * hcat(nca.data.ρc, nca.data.ρc) # initial guesses

    # Solve NCA equations
    imΣ = nlsolve(x0, method=:anderson, beta=0.9, show_trace=:true) do imΣ′
		Σ = NCA.slave_particle_Σ!(nca, imΣ′)
		_ = NCA.calculate_λ!(nca, imΣ′, Σ)
		imG = NCA.slave_particle_imG!(nca, imΣ′, Σ)
		imΣ = NCA.slave_particle_imΣ!(nca, imG)
		imΣ - imΣ′
    end.zero

    NCA.slave_particle_imG!(nca, imΣ)
end;

# ╔═╡ a5ef4166-2595-11eb-0a12-dfc7e0794078
# Fourier transform the slave-particle spectral functions
begin
	ωs = nca.mesh.mesh
	ts, _ = EchoPulse.ft(ωs, zero(nca.data.ρc), mode=-1)
	
	L = mapslices(x -> EchoPulse.ft(ωs, 1im/pi * x .* nca.data.fermi; mode=-1)[2], imG_t, dims=1)
	G = mapslices(x -> EchoPulse.ft(ωs, 1im/pi * x .* nca.data.fermi⁻; mode=-1)[2], imG_t, dims=1)
end;

# ╔═╡ a34a132c-26b0-11eb-2270-b95c8425be03
begin	
	plot(ts, L[:,1] |> imag, label="GL_b", xlabel="τ")
	plot!(ts, L[:,2] |> imag, label="GL_f")
	xlims!(-100,100)
end

# ╔═╡ df7b1ab6-25b2-11eb-3d39-69d22eee1ccc
begin	
	plot(ts, G[:,1] |> imag, label="GG_b", xlabel="τ")
	plot!(ts, G[:,2] |> imag, label="GG_f")
	xlims!(-100,100)
end

# ╔═╡ d5927c1a-25b2-11eb-1f8a-2f5e572722e7
begin
	X = repeat(imG_t[:,1], outer=[1,length(ωs)])
	
	heatmap(
		ts[1:10:end] .- ts[1], 
		ωs[1:10:end], 
		X[1:10:end,1:10:end] |> real, xlabel="T", ylabel="ω")
end

# ╔═╡ e76a6498-25b2-11eb-1172-1773740ccf31
begin
	Y = repeat(L[:,2] |> imag, outer=[1,length(ωs)])
	
	heatmap(
		ts[1:10:end] .- ts[1],
		2ts[1:10:end],
		Y[1:10:end,1:10:end] |> real, xlabel="T", ylabel="τ")
end

# ╔═╡ edae77b8-25b2-11eb-3859-f338370be8ba
begin
	Z = EchoPulse.G∞(L[:,1])
	
	heatmap(
		ts[1:length(ts)÷2][1:10:end] .+ ts[1],
		ts[1:length(ts)÷2][1:10:end] .+ ts[1],
		Z[1:10:end,1:10:end] |> imag, xlabel="t₁", ylabel="t₂")
end

# ╔═╡ Cell order:
# ╠═0b3c1108-2509-11eb-28c2-bb575678c04a
# ╠═20dfa5c6-2509-11eb-049a-37a4b96cea75
# ╠═a777bba2-8f9c-11eb-0d93-870827d56d7b
# ╠═a5ef4166-2595-11eb-0a12-dfc7e0794078
# ╠═a34a132c-26b0-11eb-2270-b95c8425be03
# ╠═df7b1ab6-25b2-11eb-3d39-69d22eee1ccc
# ╠═d5927c1a-25b2-11eb-1f8a-2f5e572722e7
# ╠═e76a6498-25b2-11eb-1172-1773740ccf31
# ╠═edae77b8-25b2-11eb-3859-f338370be8ba
