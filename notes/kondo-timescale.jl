### A Pluto.jl notebook ###
# v0.14.8

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
    β = 4e3
    model = PhotonAssistedSIAM(
        # siam
        V₀ = sqrt(0.1),# hybridization strength
        ε₀f = -0.4,# d-electron energy

        # quantum light
        Γ₀ = 0sqrt(0.005),  # light-matter coupling
        ω₀ = 2.0e-2, # photon energy
        Ω₀ = 2.0e-2,# frequency bandwitdh of the pulse
    )
    mesh = LinearMesh(-6.0, +6.0, 2^15)
	ρc = 2π * NCA.gaussian.(mesh, 0.0, 1.0)
    data = PhotonAssistedSIAMData(mesh, β, ρc, ρc)

    FreqNCA(; model = model, mesh = mesh, data = data, β = β)
end;

# ╔═╡ a777bba2-8f9c-11eb-0d93-870827d56d7b
# Obtain pseudo-particle Greater and Lesser functions
imG = let
	x0 = cat(
		abs2(nca.model.V₀) * hcat(-nca.data.ρc, -nca.data.ρc),
		abs2(nca.model.V₀) * hcat(-nca.data.ρc, +nca.data.ρc);
		dims = 3,
	) # initial guesses

	# Solve NCA equations
	imΣ = nlsolve(x0; method = :anderson, beta = 0.4, show_trace = :true) do imΣ′
		imG = NCA.slave_particle_imG(nca, imΣ′)
		imΣ = NCA.slave_particle_imΣ(nca, imG)
		return imΣ - imΣ′
	end.zero

	NCA.slave_particle_imG(nca, imΣ)
end;

# ╔═╡ a5ef4166-2595-11eb-0a12-dfc7e0794078
# Fourier transform the slave-particle spectral functions
begin
	Gᵈ = NCA.projector(nca, imG)[2];
	
    (ts, Gᵈt) = EchoPulse.ft(nca.mesh, 1im/pi * imag(Gᵈ); mode = -1)
end;

# ╔═╡ 94ec3f6f-2b16-4d07-bcc2-dd8397664304
# ωK = 0.5NCA.kondo_temp_from_peak(nca, Gᵈ |> imag)
ωK = 2NCA.kondo_temp(nca.mesh, nca.data.ρc ./ (2π), nca.model.ε₀f, nca.model.V₀)

# ╔═╡ bd6998b6-1eb6-4cf8-9582-ed37c66ef51a
begin
	ωK_ = ωK
	plot(nca.mesh.mesh, imag(Gᵈ); label = "Gd(ω)", xlabel = "ω")
	plot!(nca.mesh.mesh, (ω -> ωK_/pi/ (ω^2 + ωK_^2)).(nca.mesh), label="GK(ω)")
	xlims!(-1, 1)
	ylims!(0,10)
end

# ╔═╡ a34a132c-26b0-11eb-2270-b95c8425be03
begin
    plot(ts, imag(Gᵈt); label = "Gd(τ)", xlabel = "τ")
	plot!(ts, exp.(-ωK * abs.(ts)) * maximum(imag(Gᵈt)), label="GK(τ)")
    xlims!(-700, 700)
	# ylims!(0.0, 0.3)
end

# ╔═╡ Cell order:
# ╟─0b3c1108-2509-11eb-28c2-bb575678c04a
# ╠═20dfa5c6-2509-11eb-049a-37a4b96cea75
# ╟─a777bba2-8f9c-11eb-0d93-870827d56d7b
# ╟─a5ef4166-2595-11eb-0a12-dfc7e0794078
# ╠═94ec3f6f-2b16-4d07-bcc2-dd8397664304
# ╟─bd6998b6-1eb6-4cf8-9582-ed37c66ef51a
# ╟─a34a132c-26b0-11eb-2270-b95c8425be03
