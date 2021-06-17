### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ b9bc6a60-9622-11eb-3b18-9b0d614ab391
begin
    using Revise
    using DSP, NCA

    using NCA: nb, nf
    using NLsolve

    using Plots, LaTeXStrings
    Plots.gr()
end;

# ╔═╡ 36bbe400-9623-11eb-31e8-3f197386e7cc
# parameters (units of half-bandwidth)
begin
    β = 1e3  # initial temperature

    model = PhotonAssistedSIAM(
        # siam
        V₀ = sqrt(0.1),# hybridization strength
        ε₀f = -0.4,# d-electron energy

        # quantum light
        Γ₀ = sqrt(0.005),  # light-matter coupling
        ω₀ = 2.0e-2, # photon energy
        Ω₀ = 2.0e-2,# frequency bandwitdh of the pulse
    )

    # frequency mesh (increase the number of points if it fails)
    mesh = LinearMesh(-6.0, 6.0, 2^15)

    # custom bose distribution function
    f(x) = 1 / (exp(+1 * x) - 1)
    #   f(x) = 1 / (exp(+β * x) - 1)
end;

# ╔═╡ 03ada4c6-9624-11eb-376f-bffd9c9362c3
begin
    mesh_ = LinearMesh(first(mesh), last(mesh), length(mesh))
    ρc = 2π * NCA.gaussian.(mesh_, 0.0, 1.0)
    ργ = 2π * NCA.gaussian.(mesh_, +model.ω₀, model.Ω₀)

    # nca-problem
    nca_photon = FreqNCA(;
        β = β,
        mesh = mesh_,
        model = model,
        data = PhotonAssistedSIAMData(mesh, β, ρc, ργ, f),
    )

    nca_nophoton = let

        FreqNCA(;
            β = β,
            mesh = mesh_,
            model = PhotonAssistedSIAM(
                Γ₀ = 0.0,
                V₀ = model.V₀,
                ε₀f = model.ε₀f,
                ω₀ = model.ω₀,
                Ω₀ = model.Ω₀,
            ),
            data = PhotonAssistedSIAMData(mesh, β, ρc, ργ),
        )
    end
end;

# ╔═╡ 1a291798-96b6-11eb-3652-55f648271252
begin
    function solve!(nca)
        x0 = cat(
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, -nca.data.ρc),
            abs2(nca.model.V₀) * hcat(-nca.data.ρc, +nca.data.ρc),
            dims = 3,
        ) # initial guesses

        # Solve NCA equations
        imΣ = nlsolve(x0, method = :anderson, beta = 0.9, show_trace = :true) do imΣ′
            imG = NCA.slave_particle_imG(nca, imΣ′)
            imΣ = NCA.slave_particle_imΣ(nca, imG)
            imΣ - imΣ′
        end.zero

        return imΣ, NCA.projector(nca, NCA.slave_particle_imG(nca, imΣ))
    end
end;

# ╔═╡ 341da57e-9625-11eb-1ff3-2f0db2f8511a
begin
    imΣ_photon, (Gc_photon, Gd_photon) = solve!(nca_photon)
    imΣ_nophoton, (Gc_nophoton, Gd_nophoton) = solve!(nca_nophoton)
end;

# ╔═╡ 5b95fa66-9625-11eb-3473-1bf9e12b7850
begin
    plot(
        nca_photon.mesh.mesh,
        imag(Gd_photon);
        label = L"A_d ~\textrm{(photon assisted)}",
        title = L"A_d ~\textrm{spectral function}",
        xlab = L"\omega/D",
    )

    plot!(nca_photon.mesh.mesh, imag(Gd_nophoton); label = L"A_d ~\textrm{(no photon)}")

    # plot!(nca_photon.mesh.mesh, 0.5nca_photon.data.ρc;
    # 	label=L"A^0_c",
    # 	line=(0.5,:grey))

    vline!([model.ε₀f]; label = L"\varepsilon_d", line = (0.5, :dash, :green))

    vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (0.5, :dash, :orange))

    # savefig("spectra-equilibrium.pdf")
end

# ╔═╡ 10323bf2-96b3-11eb-3030-f56891a769f0
begin
    plot(
        nca_photon.mesh.mesh,
        imag(Gc_photon);
        label = L"A_c ~\textrm{(photon assisted)}",
        title = L"A_c ~\textrm{spectral function}",
        xlab = L"\omega/D",
    )

    plot!(nca_photon.mesh.mesh, imag(Gc_nophoton); label = L"A_c ~\textrm{(no photon)}")

    vline!([model.ε₀f]; label = L"\varepsilon_d", line = (0.5, :dash, :green))

    vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (0.5, :dash, :orange))
end

# ╔═╡ 4457d152-9626-11eb-19f9-ebe48f80ed2a
begin
    plot(
        nca_photon.mesh.mesh,
        -0.5 * imΣ_photon[:, 1, 1];
        label = L"\Sigma_b ~\textrm{(photon assisted)}",
        title = L"\Sigma_b ~\textrm{self-energy}",
        xlab = L"\omega/D",
    )

    plot!(
        nca_photon.mesh.mesh,
        nca_nophoton.data.fermi⁻ .* imΣ_nophoton[:, 1, 1];
        label = L"\Sigma_b ~\textrm{(no photon)}",
    )

    vline!([model.ε₀f]; label = L"\varepsilon_d", line = (0.5, :dash, :green))

    vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (0.5, :dash, :orange))
end

# ╔═╡ af082812-9626-11eb-11b9-016989864cf5
begin
    plot(
        nca_photon.mesh.mesh,
        -0.5 * imΣ_photon[:, 2, 1];
        label = L"\Sigma_f ~\textrm{(photon assisted)}",
        title = L"\Sigma_f ~\textrm{self-energy}",
        xlab = L"\omega/D",
    )

    plot!(
        nca_photon.mesh.mesh,
        nca_nophoton.data.fermi⁻ .* imΣ_nophoton[:, 2, 1];
        label = L"\Sigma_f ~\textrm{(no photon)}",
    )

    vline!([model.ε₀f]; label = L"\varepsilon_d", line = (0.5, :dash, :green))

    vline!([-model.ω₀, model.ω₀]; label = L"\omega_0", line = (0.5, :dash, :orange))
end

# ╔═╡ Cell order:
# ╟─b9bc6a60-9622-11eb-3b18-9b0d614ab391
# ╠═36bbe400-9623-11eb-31e8-3f197386e7cc
# ╟─03ada4c6-9624-11eb-376f-bffd9c9362c3
# ╟─1a291798-96b6-11eb-3652-55f648271252
# ╟─341da57e-9625-11eb-1ff3-2f0db2f8511a
# ╟─5b95fa66-9625-11eb-3473-1bf9e12b7850
# ╟─10323bf2-96b3-11eb-3030-f56891a769f0
# ╟─4457d152-9626-11eb-19f9-ebe48f80ed2a
# ╟─af082812-9626-11eb-11b9-016989864cf5
