""" ~: -A(τ)^* = A(-τ) """
~(a) = -conj(a)

""" Calculates the NCA Σ>(τ) and Σ<(τ) in equilibrium """
function calculate_Στ(model, data)
    (; 𝒢cGτ, 𝒢cLτ, GγG0τ, GγL0τ, GGτ, GLτ) = data

    V2G = @. model.V0^2 + im * model.Γ0^2 * (GγG0τ + ~GγL0τ)
    V2L = @. model.V0^2 + im * model.Γ0^2 * (GγL0τ + ~GγG0τ)

    ΣbG = @. -im * 2V2G * GGτ[:, 2] * ~𝒢cLτ
    ΣbL = @. -im * 2V2L * GLτ[:, 2] * ~𝒢cGτ
    ΣfG = @. +im * 1V2G * (GGτ[:, 1] * 𝒢cGτ - GGτ[:, 3] * ~𝒢cLτ)
    ΣfL = @. +im * 1V2L * (GLτ[:, 1] * 𝒢cLτ - GLτ[:, 3] * ~𝒢cGτ)
    ΣaG = @. +im * 2V2G * GGτ[:, 2] * 𝒢cGτ
    ΣaL = @. +im * 2V2L * GLτ[:, 2] * 𝒢cLτ

    GdGτ = @. +im * (GGτ[:, 2] * ~GLτ[:, 1] - GGτ[:, 3] * ~GLτ[:, 2])
    GdLτ = @. +im * (GLτ[:, 2] * ~GGτ[:, 1] - GLτ[:, 3] * ~GGτ[:, 2])

    TcGτ = @. V2G * GdGτ
    TcLτ = @. V2L * GdLτ

    return (; ΣGτ = hcat(ΣbG, ΣfG, ΣaG), ΣLτ = hcat(ΣbL, ΣfL, ΣaL), GdGτ, GdLτ, TcGτ, TcLτ)
end

""" Calculates the NCA Σ>(t,t') and Σ<(t,t') in non-equilibrium """
function calculate_Σt!(ts, h1, h2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, ΔcL, ΔcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG, GdL, GdG) = data

    𝒢cL = dmft ? 𝒢cL[t1, t2] : 𝒢cL(ts[t1], ts[t2])
    𝒢cG = dmft ? 𝒢cG[t1, t2] : 𝒢cG(ts[t1], ts[t2])

    GγL0 = GγL0[t1, t2] = data.GγL0_ext(ts[t1], ts[t2])
    GγG0 = GγG0[t1, t2] = data.GγG0_ext(ts[t1], ts[t2])

    GγL0 += data.GγL0_vac(ts[t1], ts[t2])
    GγG0 += data.GγG0_vac(ts[t1], ts[t2])

    V2G = model.V0^2 + im * model.Γ0^2 * (GγG0 + ~GγL0)
    V2L = model.V0^2 + im * model.Γ0^2 * (GγL0 + ~GγG0)

    ΣbG[t1, t2] = -im * 2V2G * GfG[t1, t2] * ~𝒢cL
    ΣbL[t1, t2] = -im * 2V2L * GfL[t1, t2] * ~𝒢cG
    ΣfG[t1, t2] = +im * 1V2G * (GbG[t1, t2] * 𝒢cG - GaG[t1, t2] * ~𝒢cL)
    ΣfL[t1, t2] = +im * 1V2L * (GbL[t1, t2] * 𝒢cL - GaL[t1, t2] * ~𝒢cG)
    ΣaG[t1, t2] = +im * 2V2G * GfG[t1, t2] * 𝒢cG
    ΣaL[t1, t2] = +im * 2V2L * GfL[t1, t2] * 𝒢cL

    GdG = GdG[t1, t2] = +im * (GfG[t1, t2] * GbL[t2, t1] - GaG[t1, t2] * GfL[t2, t1])
    GdL = GdL[t1, t2] = +im * (GfL[t1, t2] * GbG[t2, t1] - GaL[t1, t2] * GfG[t2, t1])

    TcG[t1, t2] = V2G * GdG
    TcL[t1, t2] = V2L * GdL
    ΔcL[t1, t2] = dmft ? data.GcL[t1, t2] + model.α1^2 * data.GcL_bath(ts[t1], ts[t2]) : 0.0
    ΔcG[t1, t2] = dmft ? data.GcG[t1, t2] + model.α1^2 * data.GcG_bath(ts[t1], ts[t2]) : 0.0

    TγL[t1, t2] = -im * 2model.Γ0^2 * (GdL * ~𝒢cG + 𝒢cL * ~GdG)
    TγG[t1, t2] = -im * 2model.Γ0^2 * (GdG * ~𝒢cL + 𝒢cG * ~GdL)
    return
end

function project(data::PhotonAssistedAIMData, model::PhotonAssistedAIM; ts, ws, dmft)
    (; GdL, GdG, 𝒢cL, 𝒢cG, GcL, GcG, GγL0, GγG0, TcL, TcG, TγL, TγG) = data

    # Pre-compute expensive convolution weight matrix from kbsolve!'s weights
    dts = reduce(hcat, [[w; zeros(length(ts) - length(w))] for w in ws.ws])

    # d-electron
    Gd = TimeOrderedGreenFunction(GdL.data, GdG.data, dts)

    # c-electron t-matrix
    Tc = TimeOrderedGreenFunction(TcL.data, TcG.data, dts)

    # free c-electron
    𝒢c = dmft ? TimeOrderedGreenFunction(𝒢cL.data, 𝒢cG.data, dts) : TimeOrderedGreenFunction(𝒢cL.(ts', ts), 𝒢cG.(ts', ts), dts)

    # full c-electron
    Gc = dmft ? TimeOrderedGreenFunction(GcL.data, GcG.data, dts) : 𝒢c + 𝒢c * Tc * 𝒢c

    # photon t-matrix
    Tγ = TimeOrderedGreenFunction(TγL.data, TγG.data, dts)

    # free photon
    Gγ0 = TimeOrderedGreenFunction(GγL0.data, GγG0.data, dts)

    # full photon
    Gγ = Gγ0 + Gγ0 * Tγ * Gγ0

    Gγ0_vac = TimeOrderedGreenFunction(data.GγL0_vac.(ts', ts), data.GγG0_vac.(ts', ts), dts)
    Gγ_vac = Gγ0_vac + Gγ0_vac * Tγ * Gγ0_vac

    (; Gd, Gc, 𝒢c, Gγ0, Gγ, Gγ_vac)
end
