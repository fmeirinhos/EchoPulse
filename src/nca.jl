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

function fv!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
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

    if dmft
        out[7] = -im * (∫dt1(GcG, GcL, 𝒢cL) - ∫dt2(GcL, 𝒢cG, 𝒢cL))
        out[8]= -im * (∫dt1(GcG, GcL, 𝒢cG) - ∫dt2(GcG, 𝒢cG, 𝒢cL))

        out[9]= -im * (∫dt1(TcG, TcL, 𝒢cL) - ∫dt2(TcL, 𝒢cG, 𝒢cL) + ∫dt1(GcG, GcL, GcL) - ∫dt2(GcL, GcG, GcL))
        out[10]= -im * (∫dt1(TcG, TcL, 𝒢cG) - ∫dt2(TcG, 𝒢cG, 𝒢cL) + ∫dt1(GcG, GcL, GcG) - ∫dt2(GcG, GcG, GcL))
    end

    return out
end

function fd!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    fv!(out, ts, w1, w2, t1, t2, model, data, dmft = dmft)
    @. out -= conj(out)
end

""" Calculates the NCA Σ>(t,t') and Σ<(t,t') in non-equilibrium """
function calculate_Σt!(ts, h1, h2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG, GdL, GdG) = data

    𝒢cL = dmft ? 𝒢cL[t1, t2] : 𝒢cL(ts[t1], ts[t2])
    𝒢cG = dmft ? 𝒢cG[t1, t2] : 𝒢cG(ts[t1], ts[t2])

    GγL0[t1, t2] = data.GγL0_vac(ts[t1], ts[t2]) + data.GγL0_ext(ts[t1], ts[t2])
    GγG0[t1, t2] = data.GγG0_vac(ts[t1], ts[t2]) + data.GγG0_ext(ts[t1], ts[t2])

    V2G = model.V0^2 + im * model.Γ0^2 * (GγG0[t1, t2] + GγL0[t2, t1])
    V2L = model.V0^2 + im * model.Γ0^2 * (GγL0[t1, t2] + GγG0[t2, t1])

    ΣbG[t1, t2] = -im * 2V2G * GfG[t1, t2] * ~𝒢cL
    ΣbL[t1, t2] = -im * 2V2L * GfL[t1, t2] * ~𝒢cG
    ΣfG[t1, t2] = +im * 1V2G * (GbG[t1, t2] * 𝒢cG - GaG[t1, t2] * ~𝒢cL)
    ΣfL[t1, t2] = +im * 1V2L * (GbL[t1, t2] * 𝒢cL - GaL[t1, t2] * ~𝒢cG)
    ΣaG[t1, t2] = +im * 2V2G * GfG[t1, t2] * 𝒢cG
    ΣaL[t1, t2] = +im * 2V2L * GfL[t1, t2] * 𝒢cL

    GdL = GdL[t1,t2] = +im * (GfL[t1, t2] * GbG[t2, t1] - GaL[t1, t2] * GfG[t2, t1])
    GdG = GdG[t1,t2] = +im * (GfG[t1, t2] * GbL[t2, t1] - GaG[t1, t2] * GfL[t2, t1])

    TcL[t1, t2] = V2L * GdL
    TcG[t1, t2] = V2G * GdG

    TγL[t1, t2] = -im * 2model.Γ0^2 * (GdL * ~𝒢cG + 𝒢cL * ~GdG)
    TγG[t1, t2] = -im * 2model.Γ0^2 * (GdG * ~𝒢cL + 𝒢cG * ~GdL)
    return
end

function resize_Σ!(data::PhotonAssistedAIMData; dmft, resize = false)
    if (n = size(data.GfL, 2)) > size(data.ΣfL, 2) || resize
        resize!(data.ΣaL, n)
        resize!(data.ΣaG, n)
        resize!(data.ΣfL, n)
        resize!(data.ΣfG, n)
        resize!(data.ΣbL, n)
        resize!(data.ΣbG, n)

        resize!(data.GγL0, n)
        resize!(data.GγG0, n)
        resize!(data.TγL, n)
        resize!(data.TγG, n)
        
        resize!(data.TcL, n)
        resize!(data.TcG, n)

        resize!(data.GdL, n)
        resize!(data.GdG, n)
    end
end

function initial_condition(data::PhotonAssistedAIMData; dmft)
    u0 = [data.GbL, data.GbG, data.GfL, data.GfG, data.GaL, data.GaG]
    return dmft ? [u0 ; [data.𝒢cL, data.𝒢cG, data.GcL, data.GcG]] : u0
end

function project(data::PhotonAssistedAIMData, model::PhotonAssistedAIM; ts, ws, dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG) = data

    # Pre-compute expensive convolution weight matrix from kbsolve!'s weights
    dts = reduce(hcat, [[w; zeros(length(ts) - length(w))] for w in ws.ws])

    # d-electron
    Gd = TimeOrderedGreenFunction(im * data.GfL.data .* transpose(data.GbG.data) - im * data.GaL.data .* transpose(data.GfG.data), im * data.GfG.data .* transpose(data.GbL.data) - im * data.GaG.data .* transpose(data.GfL.data), dts)

    # c-electron t-matrix
    Tc = TimeOrderedGreenFunction(data.TcL.data, data.TcG.data, dts)

    # free c-electron
    𝒢c = dmft ? TimeOrderedGreenFunction(𝒢cL.data, 𝒢cG.data, dts) : TimeOrderedGreenFunction(𝒢cL.(ts', ts), 𝒢cG.(ts', ts), dts)

    # full c-electron
    Gc = 𝒢c + 𝒢c * Tc * 𝒢c

    # photon t-matrix
    Tγ = TimeOrderedGreenFunction(TγL.data, TγG.data, dts)

    # free photon
    # G0γ = TimeOrderedGF(data.GγL0.data, data.GγG0.data, ts)
    G0γ = TimeOrderedGreenFunction(data.GγL0_ext.(ts', ts), data.GγG0_ext.(ts', ts), dts)

    # full photon
    # Gγ = TimeOrderedGreenFunction(data.GγL.data, data.GγG.data, ts)
    Gγ = G0γ + G0γ * Tγ * G0γ

    # vacuum photon
    Gγ_vac = let
        G0γ_vac = TimeOrderedGreenFunction(data.GγL0_vac.(ts', ts), data.GγG0_vac.(ts', ts), dts)
        G0γ_vac + G0γ_vac * Tγ * G0γ_vac
    end

    (; Gd, Gc, 𝒢c, G0γ, Gγ, Gγ_vac)
end

# function project(data::PhotonAssistedAIMData, model::PhotonAssistedAIM; ts, ws, dmft)
#     (; GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, GγL0, GγG0, 𝒢cL, 𝒢cG, GdL, GdG) = data

#     # Pre-compute expensive convolution weight matrix from kbsolve!'s weights
#     dts = reduce(hcat, ([w; zeros(length(ts) - length(w))] for w in ws.ws))

#     # d-electron
#     Gd = TimeOrderedGreenFunction(GdL.data, GdG.data)

#     # free c-electron
#     𝒢c = dmft ? TimeOrderedGreenFunction(𝒢cL.data, 𝒢cG.data) : TimeOrderedGreenFunction(𝒢cL.(ts', ts), 𝒢cG.(ts', ts))

#     # c-electron t-matrix
#     Tc = TimeOrderedGreenFunction(TcL.data, TcG.data)

#     # full c-electron
#     Gc = 𝒢c + TimeOrderedConvolution((𝒢c, Tc, 𝒢c), dts)

#     # free photon
#     G0γ = TimeOrderedGreenFunction(data.GγL0_f.(ts', ts), data.GγG0_f.(ts', ts)) # TimeOrderedGF(data.GγL0.data, data.GγG0.data, ts)

#     # photon t-matrix
#     Tγ = TimeOrderedGreenFunction(TγL.data, TγG.data)
#     # Tγ = let
#     #     Gc_ = 𝒢c + TimeOrderedConvolution((𝒢c, TimeOrderedGreenFunction(model.V0^2 * lesser(Gd), model.V0^2 * greater(Gd)), 𝒢c), dts)
#     #     TimeOrderedGreenFunction(-im * model.Γ0^2 * (lesser(Gd) .* transpose(greater(Gc_)) + lesser(Gc_) .* transpose(greater(Gd))), -im * model.Γ0^2 * (greater(Gd) .* transpose(lesser(Gc_)) + greater(Gc_) .* transpose(lesser(Gd))))
#     # end

#     # full photon
#     Gγ = G0γ + TimeOrderedConvolution((G0γ, Tγ, G0γ), dts) # TimeOrderedGreenFunction(data.GγL.data, data.GγG.data, ts)

#     # vacuum photon
#     Gγ_vac = let
#         G0γ_vac = TimeOrderedGreenFunction(data.GγL0_f.(1e4 .+ ts', 1e4 .+ ts), data.GγG0_f.(1e4 .+ ts', 1e4 .+ ts))
#         G0γ_vac + TimeOrderedConvolution((G0γ_vac, Tγ, G0γ_vac), dts)
#     end

#     (; Gd, Gc, 𝒢c, G0γ, Gγ, Gγ_vac)
# end

function f1!(out, ts, w1, t, model::PhotonAssistedAIM, data::PhotonAssistedAIMData)
    (; ΣbG, ΣfG) = data
    (; GbML, GbM, GfML, GfM, ΣbML, ΣbM, ΣfML, ΣfM) = data.mixed

    ∫dt(A, B) = sum(w1[i] * A[t, i] * B[i, :] for i in eachindex(w1))

    # light boson
    out[1] = -im * (-model.ε0f * GbML[t] + ∫dt(ΣbG, GbML) + ΣbML[t] .* GbM[1] + conj(ΣbM[t]) .* GbML[1])
    out[2] = -im * (-model.ε0f * GbM[t] + ∫dt(ΣbG, GbM) + ΣbM[t] .* GbM[1])

    # pseudo fermion
    out[3] = -im * (∫dt(ΣfG, GfML) + ΣfML[t] .* GfM[1] + conj(ΣfM[t]) .* GfML[1])
    out[4] = -im * (∫dt(ΣfG, GfM) + ΣfM[t] .* GfM[1])

    return out
end

function calculate_Σ1!(ts, t, model::PhotonAssistedAIM, data::PhotonAssistedAIMData)
    (; GbML, GbM, GfML, GfM) = data.mixed

    GfGτ = ft(ωs, im * 2imag(GfM[t]); inverse = true)[2]
    GfLτ = ft(ωs, im * 1imag(GfML[t]); inverse = true)[2]
    GbGτ = ft(ωs, im * 2imag(GbM[t]); inverse = true)[2]
    GbLτ = ft(ωs, im * 1imag(GbML[t]); inverse = true)[2]
    
    ΣG1 = @. -2im * model.V0^2 * GG[:, 2] * ~𝒢cL
    ΣG2 = @. +1im * model.V0^2 * (GG[:, 1] * 𝒢cG - GG[:, 3] * ~𝒢cL)
    ΣG3 = @. +2im * model.V0^2 * GG[:, 2] * 𝒢cG
    ΣL1 = @. -2im * model.V0^2 * GL[:, 2] * ~𝒢cG
    ΣL2 = @. +1im * model.V0^2 * (GL[:, 1] * 𝒢cL - GL[:, 3] * ~𝒢cG)
    ΣL3 = @. +2im * model.V0^2 * GL[:, 2] * 𝒢cL

    # ΣRω = mapslices(x -> ft(τs, x .* Θτ; inverse = false)[2], ΣGτ; dims = 1)
    # ΣLω = mapslices(x -> ft(τs, x; inverse = false)[2], ΣLτ; dims = 1)
end
