"""
Solves the problem in equilibrium

# Parameters
    - `model::PhotonAssistedModel`: A struct with model parameters
    - `ωs::Vector`: The frequency grid
    - `β::Number`: The inverse temperature
    - `KMS::Bool`: Whether to use KMS boundary conditions or have temperature fixed by the bath(s)
"""
function solve_eq(; model::PhotonAssistedModel, ωs, β, KMS)
    @assert isodd(length(ωs)) && first(ωs) == -last(ωs) "Must hold for double `ft` to be an identity."

    # Discrete Fourier grid of ωs
    τs = KadanoffBaym.ft(ωs, ωs; inverse = true)[1]
    @assert ωs ≈ KadanoffBaym.ft(τs, τs; inverse = false)[1]

    # Fourier transformations
    ft_ω(ys) = KadanoffBaym.ft(ωs, ys; inverse = true)[2]
    ft_τ(ys) = KadanoffBaym.ft(τs, ys; inverse = false)[2]

    # Solve the system in thermal equilibrium
    (; 𝒢cLτ, 𝒢cGτ, GcGτ, GcLτ, GLτ, GGτ, GcLτ_bath, GcGτ_bath, GγLτ_bath, GγGτ_bath, Στ, GcRω, GdRω, 𝒢cωR) = let
        # Proper Heaviside
        Θτ = (τ -> τ > 0 ? 1 : τ == 0 ? 0.5 : 0.0).(τs)

        # Wigner rotation of the pulse at centre-of-mass time T = 0
        GγL0τ_ext = (τ -> GγL0_ext(+τ / 2, -τ / 2; model = model)).(τs)
        GγG0τ_ext = (τ -> GγG0_ext(+τ / 2, -τ / 2; model = model)).(τs)

        # Non-interacting vacuum photon Green functions
        GγLτ_bath = ft_ω(-im * 2π * photon_dos.(ωs; cutoff = model.cutoff) .* nb.(+ωs, β))
        GγGτ_bath = ft_ω(+im * 2π * photon_dos.(ωs; cutoff = model.cutoff) .* nb.(-ωs, β))

        # Non-interacting electronic bath Green functions
        GcLτ_bath = ft_ω(+im * 2π * bethe_dos.(ωs) .* nf.(+ωs, β))
        GcGτ_bath = ft_ω(-im * 2π * bethe_dos.(ωs) .* nf.(-ωs, β))

        # Rotation to RKA basis
        Gcω_bath = zeros(ComplexF64, length(ωs), 2, 2)
        Gcω_bath[:, 1, 1] .= ft_τ((GcGτ_bath - GcLτ_bath) .* Θτ)
        Gcω_bath[:, 2, 2] .= conj(Gcω_bath[:, 1, 1])
        Gcω_bath[:, 1, 2] .= ft_τ(GcGτ_bath + GcLτ_bath)

        function eq_loop(Στ)
            # Σ>(τ) –> ΣR(ω) & Σ<(τ) –> Σ<(ω)
            ΣRω = @views mapslices(x -> ft_τ(x .* Θτ), Στ[:, 1:3]; dims = 1)
            ΣLω = @views mapslices(ft_τ, Στ[:, 4:6]; dims = 1)
            ΣcRω = @views ft_τ(Στ[:, 7] .* Θτ)
            ΣcLω = @views ft_τ(Στ[:, 8])
            ΣcKω = im * imag(2 * (ΣcLω + im * imag(ΣcRω)))

            (; GRω, GLω, 𝒢cω, Gcω) = calculate_Gω(ωs, model, (; ΣRω, ΣLω, ΣcRω, ΣcKω, Gcω_bath))

            # GR(ω) –> G>(τ) & G<(ω) –> G<(τ)
            GGτ = mapslices(x -> ft_ω(+im * 2imag(x)), GRω; dims = 1)
            GLτ = mapslices(x -> ft_ω(+im * 1imag(x)), GLω; dims = 1)

            𝒢cGτ = ft_ω(KMS ? +im * 2imag(𝒢cω[:, 1, 1]) .* nf.(-ωs, β) : im * imag(0.5 * 𝒢cω[:, 1, 2] + im * imag(𝒢cω[:, 1, 1])))
            𝒢cLτ = ft_ω(KMS ? -im * 2imag(𝒢cω[:, 1, 1]) .* nf.(+ωs, β) : im * imag(0.5 * 𝒢cω[:, 1, 2] - im * imag(𝒢cω[:, 1, 1])))

            (; ΣGτ, ΣLτ, GdGτ, GdLτ, TcGτ, TcLτ) = calculate_Στ(model, (; GGτ, GLτ, 𝒢cGτ, 𝒢cLτ, GγG0τ_ext, GγGτ_bath, GγL0τ_ext, GγLτ_bath))

            TcRω = ft_τ((TcGτ - TcLτ) .* Θτ)
            TcKω = im * imag(ft_τ(TcGτ + TcLτ))

            Σcω = similar(Gcω_bath)
            for i in eachindex(ωs)
                Tcω = [TcRω[i] TcKω[i]; 0.0 conj(TcRω[i])]
                Σcω[i, :, :] = Tcω * inv_RKA(I(2) + 𝒢cω[i, :, :] * Tcω)
            end

            Σcρτ = ft_ω(im * imag(2 * Σcω[:, 1, 1]))
            ΣcLτ = ft_ω(im * imag(0.5 * Σcω[:, 1, 2] - im * imag(Σcω[:, 1, 1])))

            # Extras
            GcGτ = ft_ω(im * imag(0.5 * Gcω[:, 1, 2] + im * imag(Gcω[:, 1, 1])))
            GcLτ = ft_ω(im * imag(0.5 * Gcω[:, 1, 2] - im * imag(Gcω[:, 1, 1])))

            GdRω = ft_τ((GdGτ - GdLτ) .* Θτ)

            (; Στ = hcat(ΣGτ, ΣLτ, Σcρτ, ΣcLτ), 𝒢cLτ, 𝒢cGτ, GcLτ, GcGτ, GLτ, GGτ, GcLτ_bath, GcGτ_bath, GγLτ_bath, GγGτ_bath, GdRω, GcRω = Gcω[:, 1, 1], 𝒢cωR = 𝒢cω[:, 1, 1])
        end

        # initial guess
        x0 = ft_ω(-im * 2π * bethe_dos.(ωs))

        # Solve equilibrium problem
        nlsolve(x -> (first ∘ eq_loop)(x) - x, hcat(x0, x0, x0, -x0, x0, x0, x0, -x0); 
            show_trace = :true, method = :anderson, iterations = 200, beta = 0.8, m = 3, ftol = 5e-12).zero |> eq_loop
    end

    # Determine the Kondo energy scale
    ωK = kondo_temperature_from_fit(ωs, GdRω, model.ε0f, model.U0)

    (; 𝒢cLτ, 𝒢cGτ, GcGτ, GcLτ, GLτ, GGτ, GcLτ_bath, GcGτ_bath, GγLτ_bath, GγGτ_bath, Στ, GcRω, 𝒢cωR, GdRω, ωs, τs, ωK, τK = inv(ωK))
end

"""
Solves the problem in non-equilibrium

# Parameters
    - `model::PhotonAssistedModel`: A struct with model parameters
    - `data::PhotonAssistedModelData`: A struct with all data required
    - `t0`: The initial time(s) (either a scalar or a vector)
    - `tmax::Number`: The maximum integration time
    - `rtol`: See KadanoffBaym.kbsolve!
    - `kmax_vie`: See KadanoffBaym.kbsolve!
"""
function solve(; model, data, t0, tmax, rtol, kmax_vie)
    # Integrate
    ts, ws = kbsolve!(
        (x...) -> fv!(x..., model, data), 
        (x...) -> fd!(x..., model, data), 
        (u0 = [data.GbL, data.GbG, data.GfL, data.GfG, data.GaL, data.GaG]; model.dmft ? [u0; [data.𝒢cL, data.𝒢cG, data.GcL, data.GcG]] : u0), (t0, tmax); 
        rtol = rtol, 
        atol = 1e-3 * rtol, 
        dtini = 1e-8, 
        kmax_vie = kmax_vie, 
        callback = (x...) -> begin
            resize_Σ!(data)
            calculate_Σt!(x..., model, data)
        end, 
        stop = (x...) -> begin
            println(" t: $(round(last(x[1]), digits=3))")
            (x[1][end] - x[1][end-1]) < 5e-12 # stops if solver gets stuck
        end)

    # Post processing
    resize_Σ!(data; resize = true)

    # Calculate observables
    (; Gd, Gc, Gγ0_ext, Gγ_ext, Gγ_vac) = project(data, model; ts = ts, ws = ws)

    (; Gd, Gc, Gγ0_ext, Gγ_ext, Gγ_vac, ts, ws)
end
