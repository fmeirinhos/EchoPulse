### A Pluto.jl notebook ###
# v0.14.8

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
    const tc = 50.0  # peak arrival time
    const Ω = 0.1 # frequency bandwidth of the pulse

    function ξ(t)
        return (Ω^2 / (2π))^(1 / 4) * exp(-(t - tc)^2 * (Ω^2 / 4))
    end

    # Photon field Hamiltonian
    ω₀ = 1.0

    # Initial field occupation
    n0 = ξ(0.0)^2 / Ω
    @assert !iszero(n0)

    # Initial conditions
    GᵞL = GreenFunction(ones(1, 1) * (-1.0im * n0), SkewHermitian)
    GᵞG = GreenFunction(ones(1, 1) * (-1.0im * (1 + n0)), SkewHermitian)

    # Pack them
    u0 = [GᵞL, GᵞG]

    # Define commutator
    ⋊(a, b) = a * b - b * adjoint(a)

    function f_vert(u, ts, t1, t2)
        GL, GG = u[1], u[2]
		
		h0 = -ω₀ - 1.0im * (ts[t1] - ts[t2]) * 0.5 * Ω^2
		h1 = 0.5im * ((tc - 0.5(ts[t1] + ts[t2])) * Ω^2)
		
        rhsL = -1.0im * (h0 * GL[t1, t2] + h1 * GL[t1, t2])
        rhsG = -1.0im * (h0 * GL[t1, t2] + h1 * GL[t1, t2])
        return [rhsL, rhsL]
    end
end

# ╔═╡ d0e7a16c-2d99-11eb-3da9-d193be8e6f43
begin
    tspan = (0.0, 100.0)
    sol = kbsolve(f_vert, (u,ts,t1) -> f_vert(u,ts,t1,t1) .- adjoint.(f_vert(u,ts,t1,t1)), u0, tspan; atol = 0, rtol = 1e-5, dtini=1e-10, dtmax=0.5)
end;

# ╔═╡ 0c6a841e-2d9a-11eb-1d8c-151d18116e55
begin
    plot(sol.t, -(imag(diag(GᵞL.data))); label = "Lesser")
    plot!(sol.t, -(imag(diag(GᵞG.data))); label = "Greater")
    plot!(
        twinx(),
        sol.t,
        ξ.(sol.t).^2;
        color = :green,
        legend = :right,
        label = "Pulse",
        line = :dot,
        xlabel = L"t",
    )
end

# ╔═╡ 26b0aed4-2d9a-11eb-127b-cbc0eabee1be
begin
    ρ, (ωs, ts) = wigner_transform_itp(GᵞG.data - GᵞL.data, sol.t)

    heatmap(ts, ωs, real(ρ); xlabel = L"$T$", ylabel = L"$\omega$")
end

# ╔═╡ 53737820-2d9a-11eb-2da7-c553a8a2c56e
# Possible analytic pulse
f(t, t′) =
    -1.0im * exp(-1.0im * (t - t′) * ω₀) * ξ(0.5 * (t + t′))^2 * exp(-(t - t′)^2 * (Ω^2 / 4))

# ╔═╡ 56f857e0-2d9a-11eb-38a9-bf4566aabc1c
heatmap(sol.t, sol.t, [imag(f(a, b)) for a in sol.t, b in sol.t])

# ╔═╡ 5bdcd6c8-2d9a-11eb-22a7-45a53df95187
heatmap(sol.t, sol.t, -(GᵞG.data) |> imag)

# ╔═╡ Cell order:
# ╠═54c8d9b4-2d99-11eb-242b-49681ab38480
# ╠═a0f8fbf2-2d99-11eb-34bf-619ba9ace9c3
# ╠═d0e7a16c-2d99-11eb-3da9-d193be8e6f43
# ╠═0c6a841e-2d9a-11eb-1d8c-151d18116e55
# ╠═26b0aed4-2d9a-11eb-127b-cbc0eabee1be
# ╠═53737820-2d9a-11eb-2da7-c553a8a2c56e
# ╠═56f857e0-2d9a-11eb-38a9-bf4566aabc1c
# ╠═5bdcd6c8-2d9a-11eb-22a7-45a53df95187
