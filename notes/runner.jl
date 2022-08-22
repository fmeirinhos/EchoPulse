### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ cb498a76-ac5b-4ac0-b250-9f8bd14508b7
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(
        m,
        Expr(
            :toplevel,
            :(eval(x) = $(Expr(:core, :eval))($name, x)),
            :(include(x) = $(Expr(:top, :include))($name, x)),
            :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
            :(include($path)),
        ),
    )
    m
end;

# ╔═╡ 3d4198f6-25ec-11ed-03ee-6fbc63159538
begin
    using Pkg
    Pkg.activate()

    using LinearAlgebra: diag
    using KadanoffBaym, LangrethsRules
    const EchoPulse = ingredients("../src/EchoPulse.jl")

    using Serialization
end;

# ╔═╡ 9f177bff-edeb-4188-9b60-192e2ebdba3b
# parameters (units of half-bandwidth)
begin
    # # inverse temperature
    # β = 1.5e2

    # model parameters
    model = EchoPulse.PhotonAssistedAIM(
        # siam
        ;
        V0 = sqrt(0.075),# hybridization strength
        ε0f = -0.31,# d-electron energy
        U0 = Inf,

        # quantum light
        Γ0 = sqrt(0.005), # light-matter coupling
        ω0 = 1.5e-2, # photon energy
        Ω0 = 1.5e-2, # photon frequency bandwidth
        α = sqrt(0.01),
    )

    # time-pulse parameters
    Nγ = 1.0e0 # maximum photon occupation
    tγ = 1.5e2 # peak arrival time

    # Bethe lattice dmft
    dmft = true

    # time-matrix initial condition
    timematrix = true

    # computational parameters
    ωs_eq = range(-9.0, 9.0; length = 2^16 - 1)
    tmax = 5.0e2
    rtol = 1e-5
end;

# ╔═╡ 2ed89c4b-563e-491c-a57b-f7f92e49d4ae
begin
    function run(str; β)
        (; data, GdRω, GcRω, 𝒢cRω, GγRω, ωK, τK, t0) = EchoPulse.PhotonAssistedAIMData(
            model = model,
            ωs = ωs_eq,
            β = β,
            tγ = tγ,
            Nγ = Nγ,
            dmft = dmft,
            timematrix = timematrix,
            nK = 2.0,
        )

        # Time-step
        @time ts, ws = kbsolve!(
            (x...) -> EchoPulse.fv!(x..., model, data; dmft = dmft),
            (x...) -> EchoPulse.fd!(x..., model, data; dmft = dmft),
            EchoPulse.initial_condition(data; dmft = dmft),
            (t0, tmax);
            rtol = rtol,
            atol = 1e-3 * rtol,
            dtini = 1e-10,
            kmax_vie = 6,
            callback = (x...) -> begin
                EchoPulse.resize_Σ!(data; dmft = dmft)
                EchoPulse.calculate_Σt!(x..., model, data; dmft = dmft)
            end,
            stop = (x...) -> begin
                println(" t: $(round(last(x[1]), digits=3))")
                false
            end,
        )

        # Post processing
        EchoPulse.resize_Σ!(data; resize = true, dmft = dmft)

        # Calculate observables
        (; Gd, Gc, Gγ0, Gγ) = EchoPulse.project(data, model; ts = ts, ws = ws, dmft = dmft)

        # Trim data
        i1 = findmin(x -> abs(x + tmax), ts)[2]
        i2 = findmin(x -> abs(x - tmax), ts)[2]

        ts = ts[i1:i2]
        Gd = (L = lesser(Gd)[i1:i2, i1:i2], G = greater(Gd)[i1:i2, i1:i2])
        Gc = (L = lesser(Gc)[i1:i2, i1:i2], G = greater(Gc)[i1:i2, i1:i2])
        Gγ = (L = lesser(Gγ)[i1:i2, i1:i2], G = greater(Gγ)[i1:i2, i1:i2])
        Gγ0 = (L = lesser(Gγ0)[i1:i2, i1:i2], G = greater(Gγ0)[i1:i2, i1:i2])

        # Wigner transforms
        ρd_ω, (ωs, Ts) = EchoPulse.wigner_transform_itp(Gd.G - Gd.L, ts; fourier = true)
        GdL_ω, _ = EchoPulse.wigner_transform_itp(Gd.L, ts; fourier = true)

        ρc_ω, _ = EchoPulse.wigner_transform_itp(Gc.G - Gc.L, ts; fourier = true)
        GcL_ω, _ = EchoPulse.wigner_transform_itp(Gc.L, ts; fourier = true)

        ργ0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0.G - Gγ0.L, ts; fourier = true)
        GγL0_ω, _ = EchoPulse.wigner_transform_itp(Gγ0.L, ts; fourier = true)

        ργ_ω, _ = EchoPulse.wigner_transform_itp(Gγ.G - Gγ.L, ts; fourier = true)
        GγL_ω, _ = EchoPulse.wigner_transform_itp(Gγ.L, ts; fourier = true)

        serialize(
            str,
            (;
                params = (;
                    V0 = model.V0,
                    ε0f = model.ε0f,
                    U0 = model.U0,
                    Γ0 = model.Γ0,
                    ω0 = model.ω0,
                    Ω0 = model.Ω0,
                    α = model.α,
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
                    ),

                # ts = ts[i1:i2],

                # data = (;
                #     Gd = (L = lesser(Gd)[i1:i2, i1:i2], G = greater(Gd)[i1:i2, i1:i2]),
                #     Gc = (L = lesser(Gc)[i1:i2, i1:i2], G = greater(Gc)[i1:i2, i1:i2]),
                #     Gγ = (L = lesser(Gγ)[i1:i2, i1:i2], G = greater(Gγ)[i1:i2, i1:i2]),
                #     Gγ0 = (L = lesser(Gγ0)[i1:i2, i1:i2], G = greater(Gγ0)[i1:i2, i1:i2]),
                # ),
            ),
        )
    end
end;

# ╔═╡ ebb478b6-2817-468a-aa9b-29f31a6b0c81
let
    i = 0
    for β in [1.5e2, 5.0e1]
        i += 1
        j = 0
        for ε0f in [-0.31, -0.33, -0.35]
            model.ε0f = ε0f
            j += 1
            k = 0
            for Ω0 in [1.5e-2, 5e-2, 1e-1]
                model.Ω0 = Ω0
                k += 1
                l = 0
                for ω0 in [+1.5e-2, +5e-2, +1e-1, +5e-1]
                    model.ω0 = ω0
                    l += 1  
                    try
                        run("../data/run-b-$(i)-e-$(j)-o-$(k)-o-$(l).jls"; β=β)
                    catch
                        continue
                    end
                end
            end
        end
    end
end
# let
# 	i = 0
# 	for β in [1.5e2, 1.0e2, 5.0e1, 1.0e1]
# 		i += 1
#         j = 0
#         for ε0f in [-0.31, -0.33, -0.35, -0.37]
#             j += 1
#             model.ε0f = ε0f
#             k = 0
#             if i == 1 && j == 1
#                 @show "skipping i=$(i), j=$(j)"
#             else
#                 for α in [sqrt(0.0025), sqrt(0.01), sqrt(0.04)]
#                     k += 1
#                     model.α = α
#             		try
#             			run("../data/run-b-$(i)-e-$(j)-a-$(k).jls"; β=β)
#             		catch
#             			continue
#             		end
#                 end
#             end
#         end
# 	end
# end

# ╔═╡ Cell order:
# ╟─cb498a76-ac5b-4ac0-b250-9f8bd14508b7
# ╠═3d4198f6-25ec-11ed-03ee-6fbc63159538
# ╠═9f177bff-edeb-4188-9b60-192e2ebdba3b
# ╠═2ed89c4b-563e-491c-a57b-f7f92e49d4ae
# ╠═ebb478b6-2817-468a-aa9b-29f31a6b0c81
