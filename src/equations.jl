""" ~: -A(τ)^* = A(-τ) """
~(a) = -conj(a)

""" Calculates GR(ω) and G<(ω) in equilibrium """
function calculate_Gω(ωs, model, ΣR, ΣL)
    GR = zero(ΣR)
    @. GR[:, 1] = ωs + model.ε0f - ΣR[:, 1]
    @. GR[:, 2] = ωs - ΣR[:, 2]
    @. GR[:, 3] = ωs - (model.ε0f + model.U) - ΣR[:, 3]
    @. GR = inv(GR)

    GL = GR .* ΣL .* conj(GR)

    # We normalise it by Q because the fixed-point of G<(ω) is scale invariant
    # f(α G<) = α G<,  ∀α
    GL *= 2π / ((ωs[2] - ωs[1]) * sum(imag(-GL[:, 1] + 2GL[:, 2] - GL[:, 3])))

    return GR, GL
end

""" Calculates Σ>(t) and Σ<(t) in equilibrium """
function calculate_Σt(model, ρcG, ρcL, GγG, GγL, GG, GL)
    V2G = model.V0^2 .+ im * model.Γ0^2 * (GγG + ~GγL)
    V2L = model.V0^2 .+ im * model.Γ0^2 * (GγL + ~GγG)

    Σt = zeros(ComplexF64, length(GγG), 6)
    @. Σt[:, 1] = -2im * V2G * GG[:, 2] * ~ρcL
    @. Σt[:, 4] = -2im * V2L * GL[:, 2] * ~ρcG
    @. Σt[:, 2] = +1im * V2G * (GG[:, 1] * ρcG - GG[:, 3] * ~ρcL)
    @. Σt[:, 5] = +1im * V2L * (GL[:, 1] * ρcL - GL[:, 3] * ~ρcG)
    @. Σt[:, 3] = +2im * V2G * GG[:, 2] * ρcG
    @. Σt[:, 6] = +2im * V2L * GL[:, 2] * ρcL

    return Σt
end

function fv!(out, ts, w1, w2, t1, t2, model::PhotonAssistedSIAM, data::TimeData; Tγ::Float64, pulse)
    (; GfL, GfG, GbL, GbG, GaL, GaG, GγL, GγG, GγL0, GγG0, ΣfL, ΣfG, ΣbL, ΣbG, ΣaL, ΣaG, TγL, TγG, GcL, GcG, TcL, TcG, εs) = data

    ∫dt1(A, B) = sum(w1[i] * A[t1, i] * B[i, t2] for i in eachindex(w1))
    ∫dt2(A, B) = sum(w2[i] * A[t1, i] * B[i, t2] for i in eachindex(w2))

    ∫dt1(A, B, C, ε) = sum(w1[i] * (A[t1, i] - B[t1, i]) * exp(+im * ε * (ts[t1] - ts[i])) * C[i, t2] for i in eachindex(w1))
    ∫dt2(A, B, C, ε) = sum(w2[i] * A[t1, i] * exp(+im * ε * (ts[t1] - ts[i])) * (B[i, t2] - C[i, t2]) for i in eachindex(w2))

    # light boson
    out[1] = -im * (-model.ε0f * GbL[t1, t2] + ∫dt1(ΣbG, GbL) - ∫dt2(ΣbL, GbG))
    out[2] = -im * (-model.ε0f * GbG[t1, t2] + ∫dt1(ΣbG, GbG) - ∫dt2(ΣbG, GbG))

    # pseudo fermion
    out[3] = -im * (∫dt1(ΣfG, GfL) - ∫dt2(ΣfL, GfG))
    out[4] = -im * (∫dt1(ΣfG, GfG) - ∫dt2(ΣfG, GfG))

    # heavy boson
    out[5] = -im * (isinf(model.U) ? 0.0 : (model.ε0f + model.U) * GaL[t1, t2] + ∫dt1(ΣaG, GaL) - ∫dt2(ΣaL, GaG))
    out[6] = -im * (isinf(model.U) ? 0.0 : (model.ε0f + model.U) * GaG[t1, t2] + ∫dt1(ΣaG, GaG) - ∫dt2(ΣaG, GaG))

    # laser pulse
    h0 = model.ω0 - im * model.Ω0 * 2tanh(model.Ω0 * (ts[t1] - ts[t2]))
    h1 = 0.0#-im * model.Ω0 * tanh(model.Ω0 * (0.5(ts[t1] + ts[t2]) - Tγ))

    out[7] = begin
        I = 0.0#∫dt(1, t1, TγG, GγL0) - ∫dt(1, t2, TγL, GγG0) - ∫dt(t2, t1, TγL, GγL0)
        -im * ((h0 + h1) * GγL[t1, t2] + (pulse ? I : 0.0))
    end
    out[8] = begin
        I = 0.0#∫dt(t2, t1, TγG, GγG0) - ∫dt(1, t1, TγL, GγG0) + ∫dt(1, t2, TγG, GγL0)
        -im * (h0 * GγG[t1, t2] + h1 * GγL[t1, t2] + (pulse ? I : 0.0))
    end

    # c-electrons
    for i in eachindex(εs)
        out[8 + i] = -im * (∫dt1(TcG, TcL, GcL[i], εs[i]) + ∫dt2(TcL, GcL[i], GcG[i], εs[i]))
        out[8 + length(εs) + i] = -im * (∫dt1(TcG, TcL, GcG[i], εs[i]) + ∫dt2(TcG, GcL[i], GcG[i], εs[i]))
    end

    return out
end

function fd!(out, ts, w1, w2, t1, t2, model::PhotonAssistedSIAM, data::TimeData; Tγ, pulse)
    fv!(out, ts, w1, w2, t1, t2, model, data, Tγ = Tγ, pulse = pulse)
    @. out -= conj(out)
end

function Σ!(ts, h1, h2, t1, t2, model::PhotonAssistedSIAM, data::TimeData)
    (; GcL, GcG, GfL, GfG, GbL, GbG, GaL, GaG, ΣaL, ΣaG, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, dos_c, εs, GγL0, GγG0) = data

    ρcL = isempty(dos_c) ? GcL(ts[t1], ts[t2]) : (εs[2] - εs[1]) * sum(exp(-im * εs[i] * (ts[t1] - ts[t2])) * GcL[i][t1, t2] * dos_c[i] for i in eachindex(dos_c))
    ρcG = isempty(dos_c) ? GcG(ts[t1], ts[t2]) : (εs[2] - εs[1]) * sum(exp(-im * εs[i] * (ts[t1] - ts[t2])) * GcG[i][t1, t2] * dos_c[i] for i in eachindex(dos_c))

    GγL0[t1, t2] = data.GγL0_f(ts[t1], ts[t2])
    GγG0[t1, t2] = data.GγG0_f(ts[t1], ts[t2])

    V2L = model.V0^2 + im * model.Γ0^2 * (GγL0[t1, t2] + GγG0[t2, t1])
    V2G = model.V0^2 + im * model.Γ0^2 * (GγG0[t1, t2] + GγL0[t2, t1])

    ΣbG[t1, t2] = -2im * V2G * GfG[t1, t2] * ~ρcL
    ΣbL[t1, t2] = -2im * V2L * GfL[t1, t2] * ~ρcG
    ΣfG[t1, t2] = +1im * V2G * (GbG[t1, t2] * ρcG - GaG[t1, t2] * ~ρcL)
    ΣfL[t1, t2] = +1im * V2L * (GbL[t1, t2] * ρcL - GaL[t1, t2] * ~ρcG)
    ΣaG[t1, t2] = +2im * V2G * GfG[t1, t2] * ρcG
    ΣaL[t1, t2] = +2im * V2L * GfL[t1, t2] * ρcL

    GdL = im * (GfL[t1, t2] * GbG[t2, t1] - GaL[t1, t2] * GfG[t2, t1])
    GdG = im * (GfG[t1, t2] * GbL[t2, t1] - GaG[t1, t2] * GfL[t2, t1])

    TcL[t1, t2] = +model.γ * V2L * GdL
    TcG[t1, t2] = +model.γ * V2G * GdG

    TγL[t1, t2] = -model.η * model.Γ0^2 * im * (GdL * ~ρcG + ρcL * ~GdG)
    TγG[t1, t2] = -model.η * model.Γ0^2 * im * (GdG * ~ρcL + ρcG * ~GdL)

    return
end

function resize_Σ!(data::TimeData; resize = false)
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

function initial_condition(data::TimeData)
    u0 = [data.GbL, data.GbG, data.GfL, data.GfG, data.GaL, data.GaG, data.GγL, data.GγG]
    return isempty(data.dos_c) ? u0 : [u0; data.GcL; data.GcG]
end

function project(data::TimeData, model::PhotonAssistedSIAM; ts, dmft)
    # pre-compute expensive weight matrix (with k=4)
    dts = reduce(hcat, ([
        KadanoffBaym.calculate_weights(ts[1:i], 4ones(Int64, i - 1))
        zeros(length(ts) - i)
    ] for i in eachindex(ts)))

    ⋆ = (x...) -> LangrethsRules.conv(x...; dts = dts)

    # d-electron
    Gd = TimeOrderedGF(im * data.GfL.data .* transpose(data.GbG.data) - im * data.GaL.data .* transpose(data.GfG.data), im * data.GfG.data .* transpose(data.GbL.data) - im * data.GaG.data .* transpose(data.GfL.data), ts)

    # @assert all(imag(diag((2data.GfL.data - data.GbL.data - data.GaL.data))) .≈ 1)

    # free photon
    G0γ = TimeOrderedGF(data.GγL0.data, data.GγG0.data, ts)

    Gγ = TimeOrderedGF(data.GγL.data, data.GγG.data, ts)

    # c-electron t-matrix
    Tc = TimeOrderedGF(data.TcL.data, data.TcG.data, ts)

    if !dmft
        # free c-electron
        G0c = TimeOrderedGF([data.GcL(t1, t2) for t1 in ts, t2 in ts], [data.GcG(t1, t2) for t1 in ts, t2 in ts], ts)

        ρc = G0c + G0c ⋆ Tc ⋆ G0c
    else
        # free c-electron
        G0c = [TimeOrderedGF(exp.(-im * ε * (ts .- ts')) .* data.GcL[i].data, exp.(-im * ε * (ts .- ts')) .* data.GcG[i].data, ts) for (i, ε) in enumerate(data.εs)]

        # c-electron spectral function
        ρc = (data.εs[2] - data.εs[1]) * sum(data.dos_c[i] * (G0c[i] + (G0c[i] ⋆ Tc ⋆ G0c[i])) for i in eachindex(data.εs))
    end

    # photon t-matrix
    Tγ = TimeOrderedGF(-im * model.Γ0^2 * (lesser(Gd) .* transpose(greater(ρc)) + lesser(ρc) .* transpose(greater(Gd))), -im * model.Γ0^2 * (greater(Gd) .* transpose(lesser(ρc)) + greater(ρc) .* transpose(lesser(Gd))), ts)

    # photon
    Gγ_out_vac = let
        G0γ_vac = TimeOrderedGF([data.GγL0_f(1e4 + t1, 1e4 + t2) for t1 in ts, t2 in ts], [data.GγG0_f(1e4 + t1, 1e4 + t2) for t1 in ts, t2 in ts], ts)
        G0γ_vac + G0γ_vac ⋆ Tγ ⋆ G0γ_vac
    end

    Gγ_out = G0γ + G0γ ⋆ Tγ ⋆ G0γ

    (; Gd, ρc, G0γ, Gγ, Gγ_out, Gγ_out_vac)
end
