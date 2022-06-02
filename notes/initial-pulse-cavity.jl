### A Pluto.jl notebook ###
# v0.19.6

using Markdown
using InteractiveUtils

# ╔═╡ 54c8d9b4-2d99-11eb-242b-49681ab38480
begin
    using Pkg; Pkg.activate()
    using KadanoffBaym

    import SpecialFunctions: erf, erfc
    import LinearAlgebra: diag

    using Plots, LaTeXStrings
    # default(fmt = :png)
end

# ╔═╡ a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
begin
    # Gaussian pulse parameters
    const tc = 20.  # peak arrival time
    const Ω = 1.0 # frequency bandwidth of the pulse

    function ξ(t)
        return (Ω^2 / (2π))^(1/4) * exp(-(t - tc)^2 * (Ω^2 / 4))
    end

    # Pump and loss terms
    const Γ = 0.1
    λ(t) = iszero(Γ) ? 0.0 : Γ * ξ(t) / sqrt(0.5 * erfc((t - tc) * Ω / sqrt(2))) |> float
    γ(t) = λ(2tc - t)
	# γ(t) = iszero(Γ) ? 0.0 : Γ * ξ(t) / sqrt(1 - 0.5 * erfc((t - tc) * Ω / sqrt(2))) |> float

    # Photon field Hamiltonian
    h(t) = 1.0 - 0.5im * (λ(t) + γ(t))

    # Artificial width of the peak
    Σ = 0.0

    # Initial field occupation
    n0 = 0.0

    # Initial conditions
    GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
    GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

    # Pack them
    u0 = [GᵞL, GᵞG];

    # Define commutator
    ⋊(a,b) = a * b - b * adjoint(a)

    function f_vert(out, times, w1, w2, t1, t2)
        t = times[t1] #|> BigFloat

        out[1] = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GᵞL[t1,t2])
        out[2] = -1.0im * ((h(t) + 1.0im * γ(t) - 1.0im * Σ) * GᵞG[t1,t2])
        return out
    end

    function f_diag(out, times, w1, w2, t1, t2)
        t = times[t1] #|> BigFloat
        println("t: $t")

        out[1] = -1.0im * (h(t) ⋊ GᵞL[t1,t1] + 1.0im * γ(t) * (GᵞL[t1,t1] + GᵞG[t1,t1]))
        out[2] = -1.0im * (h(t) ⋊ GᵞG[t1,t1] + 1.0im * λ(t) * (GᵞG[t1,t1] - GᵞL[t1,t1]) + 2.0im * γ(t) * GᵞG[t1,t1])
		
        return out
    end
end

# ╔═╡ b5160f52-db14-4fbb-a0d2-408ffeba5311
λ(21.), γ(19.)

# ╔═╡ 1c5ee080-47a8-4d09-be18-47b73d88f15a
λ(40)

# ╔═╡ d1f2a310-d246-4fb2-a1c8-9a180979c599
γ(0)

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
begin
    tspan = (0.0,40.0)
    sol = kbsolve!(f_vert, f_diag, u0, tspan; 
        atol=1e-6, rtol=1e-3)
end;

# ╔═╡ a94b0484-5e12-4557-8e16-c35138212e38
begin
	plot(sol.t, λ.(sol.t), label="Loss term")
	plot!(sol.t, t->  0.2 * (1 + tanh(0.2(t-20))))
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

# ╔═╡ e3edf1d0-718d-407c-b78f-474ce263cba6
plot(sol.t, imag(GᵞG.data - GᵞL.data) |> diag)

# ╔═╡ 26b0aed4-2d9a-11eb-127b-cbc0eabee1be
# begin
#     using Interpolations

#     itp = interpolate((sol.t,sol.t), GᵞL.data, Gridded(Linear()))
#     time_eq = range(sol.t[1], stop=sol.t[end], length=length(sol.t))

#     ρ, (ωs, ts) = wigner_transform([itp(t1,t2) for t1=time_eq, t2=time_eq], ts=time_eq);

#     heatmap(ts, ωs, real(ρ); xlabel=L"$T$", ylabel=L"$\omega$", ylims=(-10,10))
# end

# ╔═╡ 53737820-2d9a-11eb-2da7-c553a8a2c56e
# Possible analytic pulse
f(t,t′) = -1.0im * exp(-1.0im * (t-t′) * (5.0 - 1.0im * sign(t-t′) * Σ)) * ξ(0.5 * (t+t′))

# ╔═╡ 56f857e0-2d9a-11eb-38a9-bf4566aabc1c
heatmap([f(a,b) |> real for a in sol.t, b in sol.t])

# ╔═╡ 5bdcd6c8-2d9a-11eb-22a7-45a53df95187
heatmap(GᵞL.data |> real)

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═b5160f52-db14-4fbb-a0d2-408ffeba5311
# ╠═1c5ee080-47a8-4d09-be18-47b73d88f15a
# ╠═d1f2a310-d246-4fb2-a1c8-9a180979c599
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═a94b0484-5e12-4557-8e16-c35138212e38
# ╠═00b4636a-2d9a-11eb-2e36-19d6f90eba2a
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
# ╠═e3edf1d0-718d-407c-b78f-474ce263cba6
# ╠═26b0aed4-2d9a-11eb-127b-cbc0eabee1be
# ╠═53737820-2d9a-11eb-2da7-c553a8a2c56e
# ╠═56f857e0-2d9a-11eb-38a9-bf4566aabc1c
# ╠═5bdcd6c8-2d9a-11eb-22a7-45a53df95187
