""" ~: -A(τ)^* = A(-τ) """
~(a) = -conj(a)

""" Calculates GR(ω) and G<(ω) in equilibrium """
function calculate_Gω(ωs, model, ΣR, ΣL)
    GR = zero(ΣR)
    @. GR[:, 1] = ωs + model.ε0f - ΣR[:, 1]
    @. GR[:, 2] = ωs - ΣR[:, 2]
    @. GR[:, 3] = ωs - (model.ε0f + model.U0) - ΣR[:, 3]
    @. GR = inv(GR)

    GL = GR .* ΣL .* conj(GR)

    # We normalise it by Q because the fixed-point of G<(ω) is scale invariant
    # f(α G<) = α G<,  ∀α
    GL *= 2π / ((ωs[2] - ωs[1]) * sum(imag(-GL[:, 1] + 2GL[:, 2] - GL[:, 3])))

    return GR, GL
end

""" Calculates the NCA Σ>(τ) and Σ<(τ) in equilibrium """
function calculate_Στ(model, 𝒢cG, 𝒢cL, GγG, GγL, GG, GL)
    V2G = model.V0^2 .+ im * model.Γ0^2 * (GγG + ~GγL)
    V2L = model.V0^2 .+ im * model.Γ0^2 * (GγL + ~GγG)

    ΣG1 = @. -2im * V2G * GG[:, 2] * ~𝒢cL
    ΣG2 = @. +1im * V2G * (GG[:, 1] * 𝒢cG - GG[:, 3] * ~𝒢cL)
    ΣG3 = @. +2im * V2G * GG[:, 2] * 𝒢cG
    ΣL1 = @. -2im * V2L * GL[:, 2] * ~𝒢cG
    ΣL2 = @. +1im * V2L * (GL[:, 1] * 𝒢cL - GL[:, 3] * ~𝒢cG)
    ΣL3 = @. +2im * V2L * GL[:, 2] * 𝒢cL

    return hcat(ΣG1, ΣG2, ΣG3), hcat(ΣL1, ΣL2, ΣL3)
end

function fv!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; Tγ::Float64, pulse)
    (; GfL, GfG, GbL, GbG, GaL, GaG, GγL, GγG, GγL0, GγG0, ΣfL, ΣfG, ΣbL, ΣbG, ΣaL, ΣaG, TγL, TγG, 𝒢cL, 𝒢cG, GcL, GcG, TcL, TcG) = data

    ∫dt1(A, B) = sum(w1[i] * A[t1, i] * B[i, t2] for i in eachindex(w1))
    ∫dt2(A, B) = sum(w2[i] * A[t1, i] * B[i, t2] for i in eachindex(w2))

    ∫dt1(A, B, C) = sum(w1[i] * (A[t1, i] - B[t1, i]) * C[i, t2] for i in eachindex(w1))
    ∫dt2(A, B, C) = sum(w2[i] * A[t1, i] * (B[i, t2] - C[i, t2]) for i in eachindex(w2))

    # light boson
    out[1] = -im * (-model.ε0f * GbL[t1, t2] + ∫dt1(ΣbG, GbL) - ∫dt2(ΣbL, GbG))
    out[2] = -im * (-model.ε0f * GbG[t1, t2] + ∫dt1(ΣbG, GbG) - ∫dt2(ΣbG, GbG))

    # pseudo fermion
    out[3] = -im * (∫dt1(ΣfG, GfL) - ∫dt2(ΣfL, GfG))
    out[4] = -im * (∫dt1(ΣfG, GfG) - ∫dt2(ΣfG, GfG))

    # heavy boson
    out[5] = -im * (isinf(model.U0) ? 0.0 : (model.ε0f + model.U0) * GaL[t1, t2] + ∫dt1(ΣaG, GaL) - ∫dt2(ΣaL, GaG))
    out[6] = -im * (isinf(model.U0) ? 0.0 : (model.ε0f + model.U0) * GaG[t1, t2] + ∫dt1(ΣaG, GaG) - ∫dt2(ΣaG, GaG))

    # laser pulse
    h0 = model.ω0 - im * model.Ω0 * 2tanh(model.Ω0 * (ts[t1] - ts[t2]))
    h1 = 0.0#-im * model.Ω0 * tanh(model.Ω0 * (0.5(ts[t1] + ts[t2]) - Tγ))

    out[7] = begin
        I = 0.0#∫dt(1, t1, TγG-TγL, GγL0) - ∫dt(1, t2, TγL, GγG0-GγL0)
        -im * ((h0 + h1) * GγL[t1, t2] + (pulse ? I : 0.0))
    end
    out[8] = begin
        I = 0.0#∫dt(1, t1, TγG-TγL, GγG0) - ∫dt(1, t2, TγG, GγG0-GγL0)
        -im * (h0 * GγG[t1, t2] + h1 * GγL[t1, t2] + (pulse ? I : 0.0))
    end

    out[9] = -im * (∫dt1(GcG, GcL, 𝒢cL) - ∫dt2(GcL, 𝒢cG, 𝒢cL))
    out[10]= -im * (∫dt1(GcG, GcL, 𝒢cG) - ∫dt2(GcG, 𝒢cG, 𝒢cL))

    out[11]= -im * (∫dt1(TcG, TcL, 𝒢cL) - ∫dt2(TcL, 𝒢cG, 𝒢cL) + ∫dt1(GcG, GcL, GcL) - ∫dt2(GcL, GcG, GcL))
    out[12]= -im * (∫dt1(TcG, TcL, 𝒢cG) - ∫dt2(TcG, 𝒢cG, 𝒢cL) + ∫dt1(GcG, GcL, GcG) - ∫dt2(GcG, GcG, GcL))

    return out
end

function fd!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; Tγ, pulse)
    fv!(out, ts, w1, w2, t1, t2, model, data, Tγ = Tγ, pulse = pulse)
    @. out -= conj(out)
end

""" Calculates the NCA Σ>(t,t') and Σ<(t,t') in non-equilibrium """
function calculate_Σt!(ts, h1, h2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG) = data

    𝒢cL = dmft ? 𝒢cL[t1, t2] : 𝒢cL(ts[t1], ts[t2])
    𝒢cG = dmft ? 𝒢cG[t1, t2] : 𝒢cG(ts[t1], ts[t2])

    GγL0[t1, t2] = data.GγL0_f(ts[t1], ts[t2])
    GγG0[t1, t2] = data.GγG0_f(ts[t1], ts[t2])

    V2L = model.V0^2 + im * model.Γ0^2 * (GγL0[t1, t2] + GγG0[t2, t1])
    V2G = model.V0^2 + im * model.Γ0^2 * (GγG0[t1, t2] + GγL0[t2, t1])

    ΣbG[t1, t2] = -2im * V2G * GfG[t1, t2] * ~𝒢cL
    ΣbL[t1, t2] = -2im * V2L * GfL[t1, t2] * ~𝒢cG
    ΣfG[t1, t2] = +1im * V2G * (GbG[t1, t2] * 𝒢cG - GaG[t1, t2] * ~𝒢cL)
    ΣfL[t1, t2] = +1im * V2L * (GbL[t1, t2] * 𝒢cL - GaL[t1, t2] * ~𝒢cG)
    ΣaG[t1, t2] = +2im * V2G * GfG[t1, t2] * 𝒢cG
    ΣaL[t1, t2] = +2im * V2L * GfL[t1, t2] * 𝒢cL

    GdL = im * (GfL[t1, t2] * GbG[t2, t1] - GaL[t1, t2] * GfG[t2, t1])
    GdG = im * (GfG[t1, t2] * GbL[t2, t1] - GaG[t1, t2] * GfL[t2, t1])

    TcL[t1, t2] = model.V0^2 * GdL
    TcG[t1, t2] = model.V0^2 * GdG

    TγL[t1, t2] = -model.Γ0^2 * im * (GdL * ~𝒢cG + 𝒢cL * ~GdG)
    TγG[t1, t2] = -model.Γ0^2 * im * (GdG * ~𝒢cL + 𝒢cG * ~GdL)

    return
end

function resize_Σ!(data::PhotonAssistedAIMData; resize = false)
    (; GfL, ΣfL, ΣfG, ΣbL, ΣbG, ΣaL, ΣaG, TcL, TcG, TγL, TγG, GγL0, GγG0) = data
    if (n = size(GfL, 2)) > size(ΣfL, 2) || resize
        resize!(ΣaL, n)
        resize!(ΣaG, n)
        resize!(ΣfL, n)
        resize!(ΣfG, n)
        resize!(ΣbL, n)
        resize!(ΣbG, n)
        resize!(TcL, n)
        resize!(TcG, n)
        resize!(TγL, n)
        resize!(TγG, n)
        resize!(GγL0, n)
        resize!(GγG0, n)
    end
end

function initial_condition(data::PhotonAssistedAIMData; dmft)
    u0 = [data.GbL, data.GbG, data.GfL, data.GfG, data.GaL, data.GaG, data.GγL, data.GγG]
    return dmft ? [u0 ; [data.𝒢cL, data.𝒢cG, data.GcL, data.GcG]] : u0
end

function project(data::PhotonAssistedAIMData, model::PhotonAssistedAIM; ts, ws, dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG) = data

    # Pre-compute expensive convolution matrix from kbsolve!'s weights
    ⋆ = let
        dts = reduce(hcat, ([w; zeros(length(ts) - length(w))] for w in ws.ws))
        (x...) -> LangrethsRules.conv(x...; dts = dts)
    end

    # d-electron
    Gd = TimeOrderedGF(im * data.GfL.data .* transpose(data.GbG.data) - im * data.GaL.data .* transpose(data.GfG.data), im * data.GfG.data .* transpose(data.GbL.data) - im * data.GaG.data .* transpose(data.GfL.data), ts)

    # c-electron t-matrix
    Tc = TimeOrderedGF(data.TcL.data, data.TcG.data, ts)

    # free c-electron
    𝒢c = dmft ? TimeOrderedGF(𝒢cL.data, 𝒢cG.data, ts) : TimeOrderedGF(𝒢cL.(ts', ts), 𝒢cG.(ts', ts), ts)

    # full c-electron
    Gc = 𝒢c + 𝒢c ⋆ Tc ⋆ 𝒢c

    # photon t-matrix
    Tγ = TimeOrderedGF(-im * model.Γ0^2 * (lesser(Gd) .* transpose(greater(Gc)) + lesser(Gc) .* transpose(greater(Gd))), -im * model.Γ0^2 * (greater(Gd) .* transpose(lesser(Gc)) + greater(Gc) .* transpose(lesser(Gd))), ts)

    # free photon
    # G0γ = TimeOrderedGF(data.GγL0.data, data.GγG0.data, ts)
    G0γ = TimeOrderedGF(data.GγL0_f.(ts', ts), data.GγG0_f.(ts', ts), ts)

    # full photon
    # Gγ = TimeOrderedGF(data.GγL.data, data.GγG.data, ts)
    Gγ = G0γ + G0γ ⋆ Tγ ⋆ G0γ

    # vacuum photon
    Gγ_vac = let
        G0γ_vac = TimeOrderedGF(data.GγL0_f.(1e4 .+ ts', 1e4 .+ ts), data.GγG0_f.(1e4 .+ ts', 1e4 .+ ts), ts)
        G0γ_vac + G0γ_vac ⋆ Tγ ⋆ G0γ_vac
    end

    (; Gd, Gc, 𝒢c, G0γ, Gγ, Gγ_vac)
end
