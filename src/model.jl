"""
    TimeData

Data needed to solve the Kadanoff-Baym equations in the real-time contour
"""
struct TimeData{T}
    GcL::Any
    GcG::Any
    ρc::Union{Nothing,Vector{Float64}}
    ωc::Union{Nothing,Vector{Float64}}

    GfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    GγL0::Function
    GγG0::Function

    ΣfL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣfG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    ΣbG::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcL::GreenFunction{T,2,Array{T,2},SkewHermitian}
    TcG::GreenFunction{T,2,Array{T,2},SkewHermitian}
end

function TimeData(; model::PhotonAssistedSIAM, mesh, β, timematrix, ωᶜ, T₀, Nᵧ)
    # Solve the problem in equilibrium
    nca = let
        mesh = LinearMesh(first(mesh), last(mesh), length(mesh))

        # c-electron density of states
        ρc = 2π * gaussian.(mesh, 0.0, 1.0)

        # photon density of states
        ργ = 2π * gaussian.(mesh, +model.ω₀, model.Ω₀)

        FreqNCA(; 
            model = PhotonAssistedSIAM(
                V₀ = model.V₀,
                ε₀f = model.ε₀f,
                Γ₀ = zero(model.Γ₀),
                ω₀ = model.ω₀,
                Ω₀ = model.Ω₀),
            mesh = mesh, 
            data = PhotonAssistedSIAMData(mesh, β, ρc, ργ), 
            β = β)
    end

    # Obtain pseudo-particle Greater and Lesser functions in equilibrium
    imG = let
        # initial guesses
        x0 = cat(
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, -nca.data.ρc),
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, +nca.data.ρc);
            dims = 3,
        )

        # Solve NCA equations
        imΣ = fixed_point(x0) do imΣ′
            imG = NCA.slave_particle_imG(nca, imΣ′)
            imΣ = NCA.slave_particle_imΣ(nca, imG)

            # poor-man's DMFT
            if !isnothing(ωᶜ)
                nca.data.ρc = let
                  @show NCA.integrate(nca.mesh, nca.data.ρc)
                    
                  # Interpolate into a coarser mesh
                  Σc = 0.3 * nca.model.V₀^2 * NCA.projector(nca, imG)[2]
                  Σc_itp = interpolate((nca.mesh.mesh, ), Σc, Gridded(Linear()))
                  mesh_, Σc_ = sample_function(Σc_itp, range(nca.mesh[1], nca.mesh[end], length=8); tol=1e-3)
                    
                  g = similar(Σc_)

                  for (i, ωᵢ) in enumerate(mesh_)
                      g[i] = quadgk(ε -> gaussian(tan(ε) * imag(Σc_[i]) + ωᵢ - real(Σc_[i]), 0.0, 1.0), -pi / 2, pi / 2)[1]
                  end

                  # Interpolate into the nca mesh
                  itp = interpolate((mesh_, ), 2g, Gridded(Linear()))
                  g = itp.(nca.mesh.mesh)
                end
            end
            
            return imΣ
        end

        NCA.slave_particle_imG(nca, imΣ)
    end

    @unpack mesh, data = nca

    # Can compute ∑ₖ Gᶜₖ(τ)
    if isnothing(ωᶜ)
        GᶜL = let
            L = ft(mesh, im / (2π) * data.ρc .* data.fermi; mode = -1)
            itp = extrapolate(interpolate((L[1],), +L[2], Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        GᶜG = let
            G = ft(mesh, im / (2π) * data.ρc .* data.fermi⁻; mode = -1)
            itp = extrapolate(interpolate((G[1],), -G[2], Gridded(Linear())), 0)
            (t, t′) -> itp(t - t′)
        end
        ρᶜ = nothing

        # Consider many c-electron states
    elseif timematrix
        ts = ft(mesh, zero.(mesh); mode = -1)[1]

        GᶜL = [GreenFunction(G∞(+1.0im * nf(+ω, nca.β) .* exp.(-1.0im * ω * ts)), SkewHermitian) for ω in ωᶜ]
        GᶜG = [GreenFunction(G∞(-1.0im * nf(-ω, nca.β) .* exp.(-1.0im * ω * ts)), SkewHermitian) for ω in ωᶜ]
        ρᶜ = gaussian.(ωᶜ, 0.0, 1.0) .* (ωᶜ[2] - ωᶜ[1])
    else
        # Σᶜ = nca.model.V₀^2 * NCA.projector(nca, imG)[2]
        GᶜL = [GreenFunction(+1.0im * ones(1, 1) * nf(+ω, nca.β), SkewHermitian) for ω in ωᶜ]
        GᶜG = [GreenFunction(-1.0im * ones(1, 1) * nf(-ω, nca.β), SkewHermitian) for ω in ωᶜ]
        # GᶜL = [GreenFunction(+1.0im * ones(1, 1) * NCA.integrate(nca.mesh, nca.data.fermi .* imag(inv.(nca.mesh.mesh .- ω .- Σᶜ)))/pi, SkewHermitian) for ω in ωᶜ]
        # GᶜG = [GreenFunction(-1.0im * ones(1, 1) * NCA.integrate(nca.mesh, nca.data.fermi⁻ .* imag(inv.(nca.mesh.mesh .- ω .- Σᶜ)))/pi, SkewHermitian) for ω in ωᶜ]
        ρᶜ = gaussian.(ωᶜ, 0.0, 1.0) .* (ωᶜ[2] - ωᶜ[1])
    end

    # gaussian pulse
    @unpack ω₀, Ω₀ = model

    # pulse shape s.t. ∫dt ξ(t)^2 = 1
    ξ(t) = (Ω₀^2 / 2π)^(1 / 4) * exp(-t^2 * Ω₀^2 / 4)

    n = integrate(mesh.mesh, data.ργ .* data.bose) / (2π)
    N̄ᵧ(T) = isnothing(Nᵧ) ? n : Nᵧ * √(2π / Ω₀^2) * ξ(T - T₀)^2

    GᵞL₀(t, t′) = -im * ξ(t - t′)^2 * exp(-1im * ω₀ * (t - t′)) * √(2π / Ω₀^2) * (0 + N̄ᵧ(0.5(t + t′)))
    GᵞG₀(t, t′) = -im * ξ(t - t′)^2 * exp(-1im * ω₀ * (t - t′)) * √(2π / Ω₀^2) * (1 + N̄ᵧ(0.5(t + t′)))

    GᵞL = GreenFunction(GᵞL₀(0.0, 0.0) * ones(1, 1), SkewHermitian)
    GᵞG = GreenFunction(GᵞG₀(0.0, 0.0) * ones(1, 1), SkewHermitian)

    # slave-particles
    Q = integrate(mesh.mesh, -imG[:, 1, 2] + 2imG[:, 2, 2]) / 2π

    if timematrix
        ts = ft(mesh, zero.(mesh); mode = -1)[1]

        # The symmetry condition L/G(τ) = -L/G(-τ)* ⟹ L/G(ω) is purely imaginary
        L = mapslices(x -> ft(mesh, 1im / 2π * x; mode = -1)[2], imG[:, :, 2] / Q; dims = 1)
        G = mapslices(x -> ft(mesh, 1im / 2π * x; mode = -1)[2], imG[:, :, 1]; dims = 1)

        GᶠL = GreenFunction(G∞(L[:, 2]), SkewHermitian)
        GᶠG = GreenFunction(G∞(G[:, 2]), SkewHermitian)

        GᵇL = GreenFunction(G∞(L[:, 1]), SkewHermitian)
        GᵇG = GreenFunction(G∞(G[:, 1]), SkewHermitian)

        t0 = ts[1:(length(ts)÷2)]
        @assert ts[1:2:end] ≈ t0 .- reverse(t0)
        @assert size(t0, 1) == size(GᶠL, 1)
    else
        ∫dω(f) = integrate(mesh.mesh, f) / (2π)

        GᶠL = GreenFunction(im * ones(1, 1) * ∫dω(imG[:, 2, 2]) / Q, SkewHermitian)
        GᶠG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        GᵇL = GreenFunction(im * ones(1, 1) * ∫dω(imG[:, 1, 2]) / Q, SkewHermitian)
        GᵇG = GreenFunction(-im * ones(1, 1), SkewHermitian)

        @assert imag(GᶠL[1, 1]) >= 0.0
        @assert imag(GᵇL[1, 1]) <= 0.0
        @assert 2imag(GᶠL[1, 1]) - imag(GᵇL[1, 1]) ≈ 1.0
        @assert GᶠG[1, 1] == -im
        @assert GᵇG[1, 1] == -im
    end

    data = TimeData(
        GᶜL,
        GᶜG,
        ρᶜ,
        ωᶜ,
        GᶠL,
        GᶠG,
        GᵇL,
        GᵇG,
        GᵞL,
        GᵞG,
        GᵞL₀,
        GᵞG₀,
        GreenFunction(zero(GᶠL.data), SkewHermitian),
        GreenFunction(zero(GᶠG.data), SkewHermitian),
        GreenFunction(zero(GᵇL.data), SkewHermitian),
        GreenFunction(zero(GᵇG.data), SkewHermitian),
        GreenFunction(zero(GᵇL.data), SkewHermitian),
        GreenFunction(zero(GᵇG.data), SkewHermitian),
    )

    # Equilibrium c- and d-electron Green function
    G_eq = NCA.projector(nca, imG)

    # Kondo temperature estimation
    ωK = NCA.kondo_temp(nca.mesh, NCA.gaussian.(nca.mesh, 0.0, 1.0), model.ε₀f, model.V₀)

    if timematrix
        for t in eachindex(t0), t′ in eachindex(t0)
            Σ_t(model, data, t0, t, t′)
        end

        return (nca, imG, G_eq, ωK, data, t0)
    else
        return (nca, imG, G_eq, ωK, data, 0.0)
    end
end

function initial_condition(data::TimeData)
    u0 = [data.GfL, data.GfG, data.GbL, data.GbG]
    return isnothing(data.ωc) ? u0 : [u0; data.GcL; data.GcG]
end
