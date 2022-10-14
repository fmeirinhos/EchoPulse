### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ 54c8d9b4-2d99-11eb-242b-49681ab38480
begin
	import Pkg; Pkg.activate()
    using Interpolations, FFTW, KadanoffBaym, SpecialFunctions
    import LinearAlgebra: diag

    using Plots, LaTeXStrings
end

# ╔═╡ b0db48d3-b067-48a1-89ca-44c1209da047
begin
	# Photon field Hamiltonian
    ω0 = 1.0
	
    # Gaussian pulse parameters
    const tc = 80.0  # peak arrival time
    const Ω = 0.2 # frequency bandwidth of the pulse

	# Pulse shape s.t. ∫dt ξ(t)^2 = 1
	function ξ(t)
        return (Ω^2 / (2π))^(1/4) * exp(-t^2 * (Ω^2 / 4))
    end

    # Initial field occupation
    # n0 = ξ(0.0-tc)^2
    # @assert !iszero(n0)
	n0 = 10exp(-0.5 * (-tc)^2 * Ω^2)
    @assert !iszero(n0)

    # Initial conditions
    GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
    GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

    # Pack them
    u0 = [GᵞL, GᵞG]

    function f_vert!(out, ts, w1, w2, t1, t2)
		τ = ts[t1] - ts[t2]
		x = (im * ω0 - im * τ * Ω^2) / (sqrt(2) * Ω)
		hτ = ω0 - im * Ω^2 * abs(τ) + Ω * inv(sqrt(π/2) * exp(-x^2) + im * dawson(x))

		T = 1 / 2 * (ts[t1] + ts[t2])
		hT = 0.5 * (-im * (T - tc) * Ω^2)
		
		# @show h0
        out[1] = -im * (hτ * GᵞL[t1, t2] + hT * GᵞL[t1, t2])
        out[2] = -im * (hτ * GᵞG[t1, t2] + hT * GᵞL[t1, t2])
        return out
    end

	function f_diag!(out, ts, w1, w2, t1, t2)
		@show ts[end]
		f_vert!(out, ts, w1, w2, t1, t2)
		@. out -= adjoint(out)
	end
end;

# ╔═╡ a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# begin
# 	# Photon field Hamiltonian
#     ω₀ = 1.
	
#     # Gaussian pulse parameters
#     const tc = 40.0  # peak arrival time
#     const Ω = 0.1 # frequency bandwidth of the pulse

# 	# Pulse shape s.t. ∫dt ξ(t)^2 = 1
# 	function ξ(t)
#         return (Ω^2 / (2π))^(1/4) * exp(-t^2 * (Ω^2 / 4))
#     end

#     # Initial field occupation
#     n0 = ξ(0.0-tc)^2
#     @assert !iszero(n0)

#     # Initial conditions
#     GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
#     GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

#     # Pack them
#     u0 = [GᵞL, GᵞG]

#     function f_vert!(out, ts, w1, w2, t1, t2)		
# 		h0 = -ω₀ - 0.5im * Ω^2 * (ts[t1] - ts[t2]) 
# 		h1 = 0.5im * Ω^2 * (tc - 0.5(ts[t1] + ts[t2]))
		
#         out[1] = -im * (h0 * GᵞL[t1, t2] + h1 * GᵞL[t1, t2])
#         out[2] = -im * (h0 * GᵞG[t1, t2] + h1 * GᵞL[t1, t2])
#         return out
#     end

# 	function f_diag!(out, ts, w1, w2, t1, t2)
# 		@show ts[end]
# 		f_vert!(out, ts, w1, w2, t1, t2)
# 		@. out -= adjoint(out)
# 	end
# end;

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
sol = kbsolve!(f_vert!, f_diag!, u0, (0.0, 3tc);
		atol = 1e-8, rtol = 1e-4, dtini=1e-8, kmax=6,
);

# ╔═╡ 7ccb14ab-f937-45bf-b141-e34803b0beed
# begin
#     # Gaussian pulse parameters
#     const tc = 20.  # peak arrival time
#     const Ω = 1.0 # frequency bandwidth of the pulse

#     function ξ(t)
#         return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
#     end

#     # Pump and loss terms
#     const Γ = 0.1
#     λ(t) = iszero(Γ) ? 0.0 : Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2))) |> float
#     γ(t) = λ(2tc - t)
# 	# γ(t) = iszero(Γ) ? 0.0 : Γ * ξ(t) / sqrt(1 - 0.5 * erfc((t - tc) * Ω / sqrt(2))) |> float

#     # Photon field Hamiltonian
#     h(t) = 1.0 - 0.5im * (λ(t) + γ(t))

#     # Artificial width of the peak
#     Σ = 0.0

#     # Initial field occupation
#     n0 = 1e-10

#     # Initial conditions
#     GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
#     GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

#     # Pack them
#     u0 = [GᵞL, GᵞG];

#     # Define commutator
#     ⋊(a,b) = a * b - b * adjoint(a)

#     function f_vert(out, times, w1, w2, t1, t2)
#         t = times[t1] #|> BigFloat

#         out[1] = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GᵞL[t1,t2])
#         out[2] = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GᵞG[t1,t2])
#         return out
#     end

#     function f_diag(out, times, w1, w2, t1, t2)
#         t = times[t1] #|> BigFloat
#         println("t: $t")

#         out[1] = -1.0im * (h(t) ⋊ GᵞL[t1,t1] + 1.0im * γ(t) * (GᵞL[t1,t1] + GᵞG[t1,t1]))
#         out[2] = -1.0im * (h(t) ⋊ GᵞG[t1,t1] + 1.0im * λ(t) * (GᵞG[t1,t1] - GᵞL[t1,t1]) + 2.0im * γ(t) * GᵞG[t1,t1])
		
#         return out
#     end
# end

# ╔═╡ b2538f36-1b68-4927-8ba6-db66ed7f211f
# sol = kbsolve!(f_vert, f_diag, u0, (0.0, 40.); 
#         atol=1e-6, rtol=1e-3)

# ╔═╡ a53d0f5e-7bfb-42d5-9fab-b8d843e36fc9
# begin
#     plot(sol.t, λ.(sol.t), label="Loss term")
#     plot!(sol.t, γ.(sol.t), label="Pump term")
#     plot!(twinx(), sol.t, ξ.(sol.t), color=:green, legend=:right, label="Pulse", line=:dot, xlabel=L"t")
# end

# ╔═╡ 0c6a841e-2d9a-11eb-1d8c-151d18116e55
begin
    plot(sol.t, -(imag(diag(GᵞL.data))); label = "Lesser")
    plot!(sol.t, -(imag(diag(GᵞG.data))); label = "Greater")
    plot!(
        twinx(),
        sol.t,
        ξ.(sol.t .- tc).^2;
        color = :black,
        legend = :right,
        label = "Pulse",
        line = :dot,
        xlabel = L"t",
		fill = :grey
    )
end

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═b0db48d3-b067-48a1-89ca-44c1209da047
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═7ccb14ab-f937-45bf-b141-e34803b0beed
# ╠═b2538f36-1b68-4927-8ba6-db66ed7f211f
# ╠═a53d0f5e-7bfb-42d5-9fab-b8d843e36fc9
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
