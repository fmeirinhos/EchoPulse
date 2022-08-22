""" Calculates GR(ω) and G<(ω) in equilibrium """
function calculate_Gω(ωs, model, data; dmft)
    (; ΣRω, ΣLω, ΣcRω, GcRω_bath) = data

    GRω = zero(ΣRω)

    # light boson
    @. GRω[:, 1] = ωs - (-model.ε0f) - ΣRω[:, 1]

    # pseudo fermion
    @. GRω[:, 2] = ωs - ΣRω[:, 2]

    # heavy boson
    @. GRω[:, 3] = ωs - (model.ε0f + model.U0) - ΣRω[:, 3]

    # invert GF
    @. GRω = inv(GRω)
    GLω = GRω .* ΣLω .* conj(GRω)

    # We normalise it by Q because the fixed-point of G<(ω) is scale invariant
    # f(α G<) = α G<, ∀α. This way <Q> is always = 1 (apart from some \zeta prefactor)
    GLω *= 2π / ((ωs[2] - ωs[1]) * sum(imag(-GLω[:, 1] + 2GLω[:, 2] - GLω[:, 3])))

    # DMFT sum over the Bethe lattice density-of-states
    GcRω = let
        x = ωs - ΣcRω - model.α^2 * GcRω_bath
        @. 1 / 2 * (1 - sqrt(1 - 4 / x^2)) * x
    end

    # Bethe-lattice "cavity" Green function
    𝒢cRω = dmft ? inv.(ωs - GcRω - model.α^2 * GcRω_bath) : -im * π * bethe_dos.(ωs)

    return (; GRω, GLω, 𝒢cRω, GcRω)
end

function fv!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    (; GfL, GfG, GbL, GbG, GaL, GaG, ΣfL, ΣfG, ΣbL, ΣbG, ΣaL, ΣaG, 𝒢cL, 𝒢cG, GcL, GcG, ΔcL, ΔcG, TcL, TcG) = data

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
        out[7] = -im * (∫dt1(ΔcG, ΔcL, 𝒢cL) - ∫dt2(ΔcL, 𝒢cG, 𝒢cL))
        out[8] = -im * (∫dt1(ΔcG, ΔcL, 𝒢cG) - ∫dt2(ΔcG, 𝒢cG, 𝒢cL))

        out[9] = -im * (∫dt1(TcG, TcL, 𝒢cL) - ∫dt2(TcL, 𝒢cG, 𝒢cL) + ∫dt1(ΔcG, ΔcL, GcL) - ∫dt2(ΔcL, GcG, GcL))
        out[10]= -im * (∫dt1(TcG, TcL, 𝒢cG) - ∫dt2(TcG, 𝒢cG, 𝒢cL) + ∫dt1(ΔcG, ΔcL, GcG) - ∫dt2(ΔcG, GcG, GcL))
    end

    return out
end

function fd!(out, ts, w1, w2, t1, t2, model::PhotonAssistedAIM, data::PhotonAssistedAIMData; dmft)
    fv!(out, ts, w1, w2, t1, t2, model, data, dmft = dmft)
    @. out -= conj(out)
end

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
