### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 1cf881a4-ea46-11ec-3c1c-73c3deb8f624
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(m, Expr(:toplevel, :(eval(x) = $(Expr(:core, :eval))($name, x)), :(include(x) = $(Expr(:top, :include))($name, x)), :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)), :(include($path))))
    m
end;

# ╔═╡ efd2fee3-ba72-42f5-9e7f-fe32eee11a0d
begin
    using Pkg
    Pkg.activate()
    
    using LinearAlgebra: diag, LowerTriangular
    using KadanoffBaym

    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Plots, LaTeXStrings

	using Serialization
end;

# ╔═╡ fb07d4eb-95c0-497a-8fe5-8515026b5b89
begin
    # Plots.pyplot(); Plots.PyPlot.plt.style.use(["science"])
    Plots.gr()

	# Plots.default()
    Plots.default(
        fontfamily = "Computer Modern",
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        tick_direction = :in,
        minorticks = true,
        framestyle = :box,
        size = (600, 450),
        titlefontsize = 18,
        legendfontsize = 12,
        tickfontsize = 12,
        grid = true,
        #     dpi=100,
		)

    my_colors = Dict(
		:c => Plots.Colors.colorant"#0C5DA5", 
		:dω => Plots.Colors.colorant"#FF9500",
		:dτ => Plots.Colors.colorant"#00B945",
		:ωK => Plots.Colors.colorant"#9e9e9e",
		:εd => Plots.Colors.colorant"#845B97",
		:ω0 => Plots.Colors.colorant"#FF2C00"
		)
end;

# ╔═╡ 15bb008a-6b89-4cd6-aad5-5c30da31a7a5
# parameters (units of Bethe-lattice hopping)
begin
    # inverse temperature
    β = 2e2

    # model parameters
    model = EchoPulse.PhotonAssistedModel(
        # siam
        ;
        V0 = sqrt(0.08), # Hybridization strength
        ε0f = -0.35, # F-electron ground state energy
        U0 = Inf, # f-electron Coulomb repulsion

		# bath
		α1 = sqrt(0.03), # Coupling to the thermal bath
        α2 = sqrt(2.0), # Coupling to the photonic bath
		cutoff = 0.25, # Photonic bath cutoff

		# quantum light
        Γ0 = sqrt(0.0008), # Photon hybridization strength
        ω0 = 5.0e-3, # Photon central frequency
        Ω0 = 8.0e-2, # Photon spectral width
		Nγ = 5.0e0, # Photon pulse maximum average occupation
		tγ = 2.5e2, # Photon pulse peak time

    	dmft = true # Bethe lattice dmft
    )

    # maximum integration time parameters
    tmax = 1.0e3
end;

# ╔═╡ 37eaf255-9d32-48e1-bb1a-f73394ae754c
@time data_eq = EchoPulse.solve_eq(
	model = model, 
	ωs = range(-14.0, 14.0; length = 2^14 - 1), 
	β = β, 
	KMS = true);

# ╔═╡ 21895ece-5cf4-4f6f-b851-44d4fefb1788
let
    cecu6_half_bandwidth = 2.0 # eV
    cdos_half_bandwidth = 2.0
    to_eV = cecu6_half_bandwidth / cdos_half_bandwidth

    ω_to_eV(ω) = round(ω * to_eV; digits = 3)
    ω_to_THz(ω) = round(ω * to_eV * 2.4179893e2; digits = 3)
    ω_to_ps(ω) = round(inv(ω * to_eV * 2.4179893e2); digits = 3)
    ω_to_K(ω) = round(ω * to_eV * 1.1604518e4; digits = 3)

    md"""
    #### Summary
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the electronic parameters are
    1. ``\varepsilon_0^f =`` $(ω_to_eV(model.ε0f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(model.V0)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(data_eq.ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(data_eq.ωK)) ``ps`` (Kondo temperature & timescale)
    and the bath parameters were
    4. ``T = `` $(ω_to_K(inv(β)))  ``K`` (system temperature)
    5. ``\alpha_1 = `` $(ω_to_eV(model.α1)) ``eV`` (fermionic bath strength)
    6. ``\alpha_2 = `` $(ω_to_eV(model.Γ0 * model.α2)) ``eV`` (bosonic bath strength)
    7. ``\omega_\mathrm{cutoff} = `` $(ω_to_THz(model.cutoff)) ``THz`` (soft cutoff frequency of bosonic modes)
    and the gaussian pulse parameters were
	
    8. ``\omega_0 = `` $(ω_to_THz(model.ω0)) ``THz`` (pulse energy)
    9. ``\Omega_0 = `` $(ω_to_THz(model.Ω0)) ``THz``  (pulse bandwidth: $(ω_to_ps(model.Ω0/6)) ``ps``)
    10. ``\Gamma_0 = `` $(ω_to_eV(model.Γ0)) ``eV`` (light-matter coupling)
    11. ``N_\gamma = `` $(round(model.Nγ, digits=3)) (number of photons)
	
    The system will be integrated in the interval (0, $(ω_to_ps(inv(tmax)))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(model.tγ))) ``ps``.
    """
end

# ╔═╡ 1ecc4622-b311-47bd-8ba0-d6effa01fa35
let
	Plots.gr()
	
    plot(data_eq.ωs, -2imag(data_eq.GcRω); 
		label = L"-2\,\mathrm{Im}\,\sum_k G^R_{c_k}(\omega)", 
		line = (my_colors[:c], ),
		xlabel = L"\omega/t", 
		title = "Equilibrium spectral functions", 
		xlims = (-3.0, 3.0), 
		ylims = (0.0, Inf)
	)

    plot!(data_eq.ωs, -2imag(data_eq.GdRω); 
		label = L"-2\,\mathrm{Im}\,G^R_d(\omega)", 
		line = (my_colors[:dω], ),
	)
	
    vline!(model.ε0f * [1.0]; 
		label = L"\varepsilon_d", 
		line = (:dot, 1.5, my_colors[:εd], 0.75)
	)

    vline!(model.ω0 * [-1.0, 1.0]; 
		label = L"\omega_0", 
		line = (:dot, 1.5, my_colors[:ω0], 0.75)
	)
end

# ╔═╡ db46311b-40b3-418f-90ca-ab30fd1d031b
@time (; data, t0) = EchoPulse.PhotonAssistedModelData(
	data_eq = data_eq, 
	model = model, 
	timematrix = true, 
	nτK = 19.0,
	rtol = 1.0e-4,
	nmax = 6200);

# ╔═╡ 6c62195c-da9e-43dc-b091-3513fb117274
@time (; Gd, Gc, Gγ0_ext, Gγ_ext, Gγ_vac, ts, ws) = EchoPulse.solve(
	data = data,
	model = model,
	t0 = t0,
	tmax = tmax,
	rtol = 1e-5,
	kmax_vie = 5
);

# ╔═╡ d290ffe2-531c-4f12-bd08-1a282716d8ad
let
	# Trim data
	i1 = findmin(x -> abs(x + tmax), ts)[2]
	i2 = findmin(x -> abs(x - tmax), ts)[2]

	ts = ts[i1:i2]
	Gd = @views (L = lesser(Gd)[i1:i2, i1:i2], G = greater(Gd)[i1:i2, i1:i2])
	Gc = @views (L = lesser(Gc)[i1:i2, i1:i2], G = greater(Gc)[i1:i2, i1:i2])
	Gγ_ext = @views (L = lesser(Gγ_ext)[i1:i2, i1:i2], G = greater(Gγ_ext)[i1:i2, i1:i2])
	Gγ0_ext = @views (L = lesser(Gγ0_ext)[i1:i2, i1:i2], G = greater(Gγ0_ext)[i1:i2, i1:i2])
	Gγ_vac = @views (L = lesser(Gγ_vac)[i1:i2, i1:i2], G = greater(Gγ_vac)[i1:i2, i1:i2])
	
	# Wigner transforms
	ρd_ω, (ωs, Ts) = EchoPulse.wigner_transform_itp(Gd.G - Gd.L, ts; fourier = true)
	GdL_ω, _ = EchoPulse.wigner_transform_itp(Gd.L, ts; fourier = true)

	ρc_ω, _ = EchoPulse.wigner_transform_itp(Gc.G - Gc.L, ts; fourier = true)
	GcL_ω, _ = EchoPulse.wigner_transform_itp(Gc.L, ts; fourier = true)

	ργ0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0_ext.G - Gγ0_ext.L, ts; fourier = true)
	GγL0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0_ext.L, ts; fourier = true)

	ργ_ω, _ = EchoPulse.wigner_transform_itp(Gγ_ext.G - Gγ_ext.L, ts; fourier = true)
	GγL_ω, _ = EchoPulse.wigner_transform_itp(Gγ_ext.L, ts; fourier = true)

	GγL_vac, _ = EchoPulse.wigner_transform_itp(Gγ_vac.L, ts; fourier = true)

	serialize(
		"../data/test0.jls",
		(;
			params = (;
				V0 = model.V0,
				ε0f = model.ε0f,
				U0 = model.U0,
				Γ0 = model.Γ0,
				ω0 = model.ω0,
				Ω0 = model.Ω0,
				α1 = model.α1,
                α2 = model.α2,
				Nγ = model.Nγ,
				tγ = model.tγ,
				cutoff = model.cutoff,
				τK = data_eq.τK,
				ωK = data_eq.ωK,
				β = β,
			),

			Ts = Ts,
			ωs = ωs,

			data = (;
				ρd_ω,
				GdL_ω,
				ρc_ω,
				GcL_ω,
				ργ0_ω,
				GγL0_ω,
				ργ_ω,
				GγL_ω,
				GγL_vac
				),
		),
	)
end

# ╔═╡ ebbdba0e-450c-4427-b626-6bf66afe2b37
GC.gc()

# ╔═╡ 500d67f8-4893-4d5b-82b9-ea009eda2b18
varinfo(@__MODULE__, imported=true)

# ╔═╡ Cell order:
# ╟─1cf881a4-ea46-11ec-3c1c-73c3deb8f624
# ╠═efd2fee3-ba72-42f5-9e7f-fe32eee11a0d
# ╟─fb07d4eb-95c0-497a-8fe5-8515026b5b89
# ╠═15bb008a-6b89-4cd6-aad5-5c30da31a7a5
# ╠═37eaf255-9d32-48e1-bb1a-f73394ae754c
# ╟─21895ece-5cf4-4f6f-b851-44d4fefb1788
# ╟─1ecc4622-b311-47bd-8ba0-d6effa01fa35
# ╠═db46311b-40b3-418f-90ca-ab30fd1d031b
# ╠═6c62195c-da9e-43dc-b091-3513fb117274
# ╟─d290ffe2-531c-4f12-bd08-1a282716d8ad
# ╠═ebbdba0e-450c-4427-b626-6bf66afe2b37
# ╠═500d67f8-4893-4d5b-82b9-ea009eda2b18
