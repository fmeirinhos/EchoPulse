### A Pluto.jl notebook ###
# v0.19.6

using Markdown
using InteractiveUtils

# ╔═╡ 54c8d9b4-2d99-11eb-242b-49681ab38480
begin
	import Pkg; Pkg.activate()
    using Interpolations, FFTW, KadanoffBaym
    import LinearAlgebra: diag

    using Plots, LaTeXStrings
    # default(fmt = :png)
end

# ╔═╡ 2f5f48b6-53c6-4a33-a71a-f8fd038b07b0
using SpecialFunctions

# ╔═╡ a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
begin
	# Photon field Hamiltonian
    ω₀ = 1.
	
    # Gaussian pulse parameters
    const tc = 100.0  # peak arrival time
    const Ω = 0.3 # frequency bandwidth of the pulse

	# Pulse shape s.t. ∫dt ξ(t)^2 = 1
	function ξ(t)
        return (Ω^2 / (2π))^(1/4) * exp(-t^2 * (Ω^2 / 4))
    end

    # Initial field occupation
    n0 = ξ(0.0-tc)^2
    # @assert !iszero(n0)

    # Initial conditions
    GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
    GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

    # Pack them
    u0 = [GᵞL, GᵞG]

    function f_vert!(out, ts, w1, w2, t1, t2)		
		h0 = -ω₀ - 0.5im * Ω^2 * (ts[t1] - ts[t2]) 
		h1 = 0.5im * Ω^2 * (tc - 0.5(ts[t1] + ts[t2]))
		
        out[1] = -im * (h0 * GᵞL[t1, t2] + h1 * GᵞL[t1, t2])
        out[2] = -im * (h0 * GᵞL[t1, t2] + h1 * GᵞL[t1, t2])
        return out
    end

	function f_diag!(out, ts, w1, w2, t1, t2)
		f_vert!(out, ts, w1, w2, t1, t2)
		@. out -= adjoint(out)
	end
end;

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
sol = kbsolve!(f_vert!, f_diag!, u0, (0.0, 2tc);
		atol = 0, rtol = 1e-3, dtini=1e-10, dtmax=0.5
);

# ╔═╡ 0c6a841e-2d9a-11eb-1d8c-151d18116e55
begin
    plot(sol.t, -(imag(diag(GᵞL.data))); label = "Lesser")
    plot!(sol.t, -(imag(diag(GᵞG.data))); label = "Greater")
    plot!(
        twinx(),
        sol.t,
        ξ.(sol.t .- tc);
        color = :green,
        legend = :right,
        label = "Pulse",
        line = :dot,
        xlabel = L"t",
    )
end

# ╔═╡ 53737820-2d9a-11eb-2da7-c553a8a2c56e
begin
	# Analytic pulse
	Ω_(t, t′) = Ω#0.5 * (t + t′) < tc ? Ω * 0.5 * (t + t′) : Ω * (2tc - 0.5 * (t + t′))
	
	gl(t, t′) = -im * exp(-1.0im * (t - t′) * ω₀) * ξ(0.5 * (t + t′) - tc)^2 * exp(-(t - t′)^2 * (Ω_(t, t′)^2 / 16))
	
	gg(t, t′) = -im * exp(-1.0im * (t - t′) * ω₀) * (1 + ξ(0.5 * (t + t′) - tc)^2) * exp(-(t - t′)^2 * (Ω_(t, t′)^2 / 16))

	ρ(t, t′) = gg(t, t′) - gl(t, t′)
end

# ╔═╡ 88558db4-c7bf-41cc-855f-fefa64c145dc
begin
	α=0.1
	λ(t) = α * ξ(t) / sqrt(0.5 * erfc((t) * Ω / sqrt(2))) |> float
	γ(t) = α * ξ(t)#λ(-t)
end

# ╔═╡ 4d88f11a-9654-45a1-92fb-d2c732955a94
begin
	function f_vert2(out, ts, w1, w2, t1, t2)
		t = ts[t1] - tc
		
		out[1] = -im * (+ω₀ +im/2 * (γ(t) - λ(t))) * GᵞL[t1, t2]
		out[2] = -im * (+ω₀ +im/2 * (γ(t) - λ(t))) * GᵞG[t1, t2]
		
		return out
	end
	
	function f_diag2(out, ts, w1, w2, t1, t2)
		t = ts[t1] - tc

		println(" t: $t")
		
		out[1] = γ(t) * GᵞG[t1, t2] - λ(t) * GᵞL[t1, t2]
		out[2] = γ(t) * GᵞG[t1, t2] - λ(t) * GᵞL[t1, t2]
		return out
	end

	sol2 = kbsolve!(f_vert2, f_diag2, u0, (0.0, 2tc); 
		atol = 0, rtol = 1e-5, dtini=1e-13)
end

# ╔═╡ a84f3b20-b4b9-4e3e-9a3b-f15bbaa0b0ad
plot(sol2.t, t -> Ω_(t, t))

# ╔═╡ 90714231-8592-4bb0-beaa-37c943d15aea
plot(sol2.t, t -> γ(tc - t) - λ(tc - t))

# ╔═╡ c8b01a9d-5a18-4e05-8526-1b7aac6a8bc3
(t -> γ(tc - t) - λ(tc - t))(16)

# ╔═╡ 60f6505d-7901-48c0-b23c-aa1eccb208e9
begin
	plot(sol2.t, imag(-GᵞL.data) |> diag)
	# plot!(twinx(), sol2.t, t -> ξ(t - tc), color=:darkgreen)
end

# ╔═╡ 7b24219f-64c6-40a7-814c-bdf0a1a55731
heatmap(sol2.t, sol2.t, -ρ.(sol2.t, sol2.t') |> imag)

# ╔═╡ 47ad2ff7-a46a-4549-b387-203dffa8132b
heatmap(sol2.t, sol2.t, -imag(GᵞG.data - GᵞL.data))

# ╔═╡ 826d0443-a5ff-4c9f-b7f0-a5dfb54907f4
plot(sol2.t, imag(GᵞG.data - GᵞL.data) |> diag)

# ╔═╡ 680863f1-fc34-47c5-83d5-da34ec346802
(w, (Ts, ωs)) = wigner_transform_itp(GᵞG.data - GᵞL.data, sol2.t);

# ╔═╡ d0c70775-a2f9-4f00-a3d4-81a1a8e02026
heatmap(ωs, Ts, -imag(w))

# ╔═╡ 517c83f5-0b09-42fd-b7b4-9e46b32c2d05
heatmap(sol2.t, sol2.t, -imag(GᵞG.data)+imag(GᵞL.data))

# ╔═╡ 4505419f-ac04-4f6a-984c-61c24520e72a
heatmap(sol2.t, sol2.t, -imag(GᵞG.data))

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
# ╠═53737820-2d9a-11eb-2da7-c553a8a2c56e
# ╠═a84f3b20-b4b9-4e3e-9a3b-f15bbaa0b0ad
# ╠═2f5f48b6-53c6-4a33-a71a-f8fd038b07b0
# ╠═88558db4-c7bf-41cc-855f-fefa64c145dc
# ╠═4d88f11a-9654-45a1-92fb-d2c732955a94
# ╠═90714231-8592-4bb0-beaa-37c943d15aea
# ╠═c8b01a9d-5a18-4e05-8526-1b7aac6a8bc3
# ╠═60f6505d-7901-48c0-b23c-aa1eccb208e9
# ╠═7b24219f-64c6-40a7-814c-bdf0a1a55731
# ╠═47ad2ff7-a46a-4549-b387-203dffa8132b
# ╠═826d0443-a5ff-4c9f-b7f0-a5dfb54907f4
# ╠═680863f1-fc34-47c5-83d5-da34ec346802
# ╠═d0c70775-a2f9-4f00-a3d4-81a1a8e02026
# ╠═517c83f5-0b09-42fd-b7b4-9e46b32c2d05
# ╠═4505419f-ac04-4f6a-984c-61c24520e72a
