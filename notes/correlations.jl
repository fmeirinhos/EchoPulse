### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 8e9baa64-399c-4bbe-8505-31ea274defba
using Serialization

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

    using LinearAlgebra: diag
    using KadanoffBaym, LangrethsRules

    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Plots, LaTeXStrings
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
# parameters (units of half-bandwidth)
begin
    # inverse temperature
    β = 1.5e2

    # model parameters
    model = EchoPulse.PhotonAssistedAIM(
        # siam
        ;
        V0 = sqrt(0.075),# hybridization strength
        ε0f = -0.31,# d-electron energy
        U0 = Inf,
		
        # quantum light
        Γ0 = sqrt(0.005), # light-matter coupling
        ω0 = 5.0e-2, # photon energy
        Ω0 = 3.0e-2, # photon frequency bandwidth

		α=0.0,
        α2=0.05
    )

    # time-pulse parameters
    Nγ = 1.0e0 # maximum photon occupation
    tγ = 1.0e2 # peak arrival time

    # Bethe lattice dmft
    dmft = true

    # time-matrix initial condition
    timematrix = true

    # computational parameters
	ωs_eq = range(-11., 11.; length = 2^16 - 1)
    tmax = 2.5e2
    rtol = 1e-5
end;

# ╔═╡ 75af13de-fcb5-4e1d-804f-dc3cc0809eed
begin
    (; data, GdRω, GcRω, 𝒢cRω, ωK, τK, t0) = EchoPulse.PhotonAssistedAIMData(
		model = model, 
		ωs = ωs_eq, 
		β = β, 
		tγ = tγ,
		Nγ = Nγ, 
		dmft = dmft,
		timematrix = timematrix, 
		nK = 2.0
		)
end;

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
    Assuming the bandwidth of $\textrm{CeCu}_6$ to be ``\sim`` $(2cecu6_half_bandwidth) ``eV``, the  parameters are
    1. ``\varepsilon_d =`` $(ω_to_eV(model.ε0f)) ``eV`` (``d-``orbital energy)
    2. ``V_0 =  `` $(ω_to_eV(model.V0)) ``eV`` (hybridisation strength)
    3. ``T_K = `` $(ω_to_K(ωK)) ``K``, ``\tau_K = `` $(ω_to_ps(ωK)) ``ps`` (Kondo temperature & timescale)
    4. ``T = `` $(ω_to_K(1/β))  ``K`` (system temperature)
    and the gaussian-shaped pulse parameters were
    5. ``\omega = `` $(ω_to_THz(model.ω0)) ``THz`` (pulse energy)
    6. ``\Omega = `` $(ω_to_THz(model.Ω0)) ``THz``  (pulse bandwidth: $(ω_to_ps(model.Ω0)) ``ps``)
    7. ``\Gamma = `` $(ω_to_eV(model.Γ0)) ``eV`` (light-matter coupling)
    8. ``N_\gamma = `` $(round(Nγ, digits=3)) ``n_\gamma`` (number of photons)

    The system will be integrated in the interval (0, $(ω_to_ps(inv(tmax)))) ``ps`` with the light-pulse reaching its maximum at $(ω_to_ps(inv(tγ))) ``ps``.
    """
end

# ╔═╡ 1ecc4622-b311-47bd-8ba0-d6effa01fa35
let
	Plots.gr()
	
    plot(ωs_eq, -2imag(GcRω); 
		label = L"-2\,\mathrm{Im}\,\sum_k G^R_{c_k}(\omega)", 
		line = (my_colors[:c], ),
		xlabel = L"\omega/D", 
		title = "Equilibrium spectral functions", 
		xlims = (-3.0, 3.0), 
		ylims = (0.0, Inf)
	)

    plot!(ωs_eq, -2imag(GdRω); 
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

    vspan!(ωK * [-1 / 2, 1 / 2]; 
		label = L"\omega_K",
		fill = (my_colors[:ωK], 0.3),
	)
end

# ╔═╡ d12b32d7-4101-44c9-87b1-4ed1438decf7
begin
    # Time-step
    @time ts, ws = kbsolve!(
		(x...) -> EchoPulse.fv!(x..., model, data; dmft = dmft), 
        (x...) -> EchoPulse.fd!(x..., model, data; dmft = dmft), 
        EchoPulse.initial_condition(data; dmft = dmft), 
        (t0, tmax); 
        rtol = rtol, 
        atol = 1e-3 * rtol, 
		dtini = 1e-10,
		kmax_vie=6,
        callback = (x...) -> begin
            EchoPulse.resize_Σ!(data; dmft = dmft)
            EchoPulse.calculate_Σt!(x..., model, data; dmft = dmft)
        end, 
        stop = (x...) -> begin
            println(" t: $(round(last(x[1]), digits=3))")
            false
        end)

    # Post processing
    EchoPulse.resize_Σ!(data; resize = true, dmft = dmft)
end;

# ╔═╡ 5feec3dd-f53d-4e67-8b1a-ca66131b17eb
begin
	# Calculate observables
	(; Gd, Gc, Gγ0, Gγ, Gγ_vac) = EchoPulse.project(data, model; ts = ts, ws = ws, dmft = dmft);
end;

# ╔═╡ 98522051-1ab9-4db2-9bf2-2b829b98b77a
let
	# Trim data
	i1 = findmin(x -> abs(x + tmax), ts)[2]
	i2 = findmin(x -> abs(x - tmax), ts)[2]

	ts = ts[i1:i2]
	@views Gd = (L = lesser(Gd)[i1:i2, i1:i2], G = greater(Gd)[i1:i2, i1:i2])
	@views Gc = (L = lesser(Gc)[i1:i2, i1:i2], G = greater(Gc)[i1:i2, i1:i2])
	@views Gγ = (L = lesser(Gγ)[i1:i2, i1:i2], G = greater(Gγ)[i1:i2, i1:i2])
	@views Gγ0 = (L = lesser(Gγ0)[i1:i2, i1:i2], G = greater(Gγ0)[i1:i2, i1:i2])
	@views Gγ_vac = (L = lesser(Gγ_vac)[i1:i2, i1:i2], G = greater(Gγ_vac)[i1:i2, i1:i2])
	
	# Wigner transforms
	ρd_ω, (ωs, Ts) = EchoPulse.wigner_transform_itp(Gd.G - Gd.L, ts; fourier = true)
	GdL_ω, _ = EchoPulse.wigner_transform_itp(Gd.L, ts; fourier = true)

	ρc_ω, _ = EchoPulse.wigner_transform_itp(Gc.G - Gc.L, ts; fourier = true)
	GcL_ω, _ = EchoPulse.wigner_transform_itp(Gc.L, ts; fourier = true)

	ργ0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0.G - Gγ0.L, ts; fourier = true)
	GγL0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0.L, ts; fourier = true)

	ργ_ω, _ = EchoPulse.wigner_transform_itp(Gγ.G - Gγ.L, ts; fourier = true)
	GγL_ω, _ = EchoPulse.wigner_transform_itp(Gγ.L, ts; fourier = true)

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
				α = model.α,
                α2 = model.α2,
				Nγ = Nγ,
				tγ = tγ,
				τK = τK,
				ωK = ωK,
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
# ╠═75af13de-fcb5-4e1d-804f-dc3cc0809eed
# ╟─21895ece-5cf4-4f6f-b851-44d4fefb1788
# ╟─1ecc4622-b311-47bd-8ba0-d6effa01fa35
# ╠═d12b32d7-4101-44c9-87b1-4ed1438decf7
# ╠═5feec3dd-f53d-4e67-8b1a-ca66131b17eb
# ╠═8e9baa64-399c-4bbe-8505-31ea274defba
# ╠═98522051-1ab9-4db2-9bf2-2b829b98b77a
# ╠═ebbdba0e-450c-4427-b626-6bf66afe2b37
# ╠═500d67f8-4893-4d5b-82b9-ea009eda2b18
