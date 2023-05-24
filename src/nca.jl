""" ~: -A≷(τ)^* = A≷(-τ) """
~(a) = -conj(a)

""" Calculates the NCA Σ≷(τ) in equilibrium """
function calculate_Στ(model, data)
    (; 𝒢cGτ, 𝒢cLτ, GGτ, GLτ) = data

    GγL0τ = @. data.GγL0τ_ext + model.α2^2 * data.GγLτ_bath
    GγG0τ = @. data.GγG0τ_ext + model.α2^2 * data.GγGτ_bath

    V2Gτ = @. model.V0^2 + im * model.Γ0^2 * (GγG0τ + ~GγL0τ)
    V2Lτ = @. model.V0^2 + im * model.Γ0^2 * (GγL0τ + ~GγG0τ)

    ΣbGτ = @. -im * model.N * V2Gτ * GGτ[:, 2] * ~𝒢cLτ
    ΣbLτ = @. -im * model.N * V2Lτ * GLτ[:, 2] * ~𝒢cGτ
    ΣfGτ = @. +im * V2Gτ * (GGτ[:, 1] * 𝒢cGτ - GGτ[:, 3] * ~𝒢cLτ)
    ΣfLτ = @. +im * V2Lτ * (GLτ[:, 1] * 𝒢cLτ - GLτ[:, 3] * ~𝒢cGτ)
    ΣaGτ = @. +im * model.N * V2Gτ * GGτ[:, 2] * 𝒢cGτ
    ΣaLτ = @. +im * model.N * V2Lτ * GLτ[:, 2] * 𝒢cLτ

    GdGτ = @. +im * (GGτ[:, 2] * ~GLτ[:, 1] - GGτ[:, 3] * ~GLτ[:, 2])
    GdLτ = @. +im * (GLτ[:, 2] * ~GGτ[:, 1] - GLτ[:, 3] * ~GGτ[:, 2])

    TcGτ = @. V2Gτ * GdGτ
    TcLτ = @. V2Lτ * GdLτ

    return (; ΣGτ = hcat(ΣbGτ, ΣfGτ, ΣaGτ), ΣLτ = hcat(ΣbLτ, ΣfLτ, ΣaLτ), GdGτ, GdLτ, TcGτ, TcLτ)
end

""" Calculates the NCA Σ≷(t, t') in non-equilibrium """
function calculate_Σt!(ts, h1, h2, t1, t2, model::PhotonAssistedModel, data::PhotonAssistedModelData)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, ΔcL, ΔcG, TγL, TγG, 𝒢cL, 𝒢cG, GdL, GdG) = data

    𝒢cL = model.dmft ? 𝒢cL[t1, t2] : 𝒢cL(ts[t1], ts[t2])
    𝒢cG = model.dmft ? 𝒢cG[t1, t2] : 𝒢cG(ts[t1], ts[t2])

    GγL0 = data.GγL0_ext(ts[t1], ts[t2]) + model.α2^2 * data.GγL_bath(ts[t1], ts[t2])
    GγG0 = data.GγG0_ext(ts[t1], ts[t2]) + model.α2^2 * data.GγG_bath(ts[t1], ts[t2])

    V2G = model.V0^2 + im * model.Γ0^2 * (GγG0 + ~GγL0)
    V2L = model.V0^2 + im * model.Γ0^2 * (GγL0 + ~GγG0)

    ΣbG[t1, t2] = -im * model.N * V2G * GfG[t1, t2] * ~𝒢cL
    ΣbL[t1, t2] = -im * model.N * V2L * GfL[t1, t2] * ~𝒢cG
    ΣfG[t1, t2] = +im * V2G * (GbG[t1, t2] * 𝒢cG - GaG[t1, t2] * ~𝒢cL)
    ΣfL[t1, t2] = +im * V2L * (GbL[t1, t2] * 𝒢cL - GaL[t1, t2] * ~𝒢cG)
    ΣaG[t1, t2] = +im * model.N * V2G * GfG[t1, t2] * 𝒢cG
    ΣaL[t1, t2] = +im * model.N * V2L * GfL[t1, t2] * 𝒢cL

    GdG = GdG[t1, t2] = +im * (GfG[t1, t2] * GbL[t2, t1] - GaG[t1, t2] * GfL[t2, t1])
    GdL = GdL[t1, t2] = +im * (GfL[t1, t2] * GbG[t2, t1] - GaL[t1, t2] * GfG[t2, t1])

    TcG[t1, t2] = V2G * GdG
    TcL[t1, t2] = V2L * GdL
    ΔcL[t1, t2] = model.dmft ? data.GcL[t1, t2] + model.α1^2 * data.GcL_bath(ts[t1], ts[t2]) : 0.0
    ΔcG[t1, t2] = model.dmft ? data.GcG[t1, t2] + model.α1^2 * data.GcG_bath(ts[t1], ts[t2]) : 0.0

    TγL[t1, t2] = -im * model.N * model.Γ0^2 * (GdL * ~𝒢cG + 𝒢cL * ~GdG)
    TγG[t1, t2] = -im * model.N * model.Γ0^2 * (GdG * ~𝒢cL + 𝒢cG * ~GdL)
    return
end

function project(data::PhotonAssistedModelData, model::PhotonAssistedModel; ts, ws)
    (; GdL, GdG, 𝒢cL, 𝒢cG, GcL, GcG, GγL0_ext, GγG0_ext, GγL_bath, GγG_bath, TcL, TcG, TγL, TγG) = data

    # Pre-compute expensive convolution weight matrix from kbsolve!'s weights
    dts = reduce(hcat, [[w; zeros(length(ts) - length(w))] for w in ws]) |> UpperTriangular
    ⋆(a, b) = conv(a, b, dts)

    # d-electron
    Gd = TimeOrderedGreenFunction(GdL.data, GdG.data)

    # c-electron t-matrix
    Tc = TimeOrderedGreenFunction(TcL.data, TcG.data)

    # free c-electron
    𝒢c = model.dmft ? TimeOrderedGreenFunction(𝒢cL.data, 𝒢cG.data) : TimeOrderedGreenFunction(𝒢cL.(ts, ts'), 𝒢cG.(ts, ts'))

    # full c-electron
    Gc = model.dmft ? TimeOrderedGreenFunction(GcL.data, GcG.data) : 𝒢c + 𝒢c ⋆ Tc ⋆ 𝒢c

    # photon t-matrix
    Tγ = TimeOrderedGreenFunction(TγL.data, TγG.data)

    # free photon
    Gγ0_ext = TimeOrderedGreenFunction(GγL0_ext.(ts, ts'), GγG0_ext.(ts, ts'))

    # full photon
    Gγ_ext = Gγ0_ext + Gγ0_ext ⋆ Tγ ⋆ Gγ0_ext

    Gγ0_vac = TimeOrderedGreenFunction(GγL_bath.(ts, ts'), GγG_bath.(ts, ts'))
    Gγ_vac = Gγ0_vac + Gγ0_vac ⋆ Tγ ⋆ Gγ0_vac

    (; Gd, Gc, 𝒢c, Gγ0_ext, Gγ_ext, Gγ_vac)
end
