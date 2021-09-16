function fv!(out, model::PhotonAssistedSIAM, data::TimeData, ts, t1, t2)
    @unpack GfL, GfG, GbL, GbG, ΣfL, ΣfG, ΣbL, ΣbG, GcL, GcG, TcL, TcG, ωc = data

    ∫dt(x...) = integrate(ts, t1, t2, x...)

    # pseudo fermion
    out[1] = -1.0im * (model.ε₀f * GfL[t1, t2] + ∫dt(1, t1, ΣfG, GfL) - ∫dt(1, t2, ΣfL, GfG))
    out[2] = -1.0im * (model.ε₀f * GfG[t1, t2] + ∫dt(t2, t1, ΣfG, GfG))

    # slave boson
    out[3] = -1.0im * (∫dt(1, t1, ΣbG, GbL) - ∫dt(1, t2, ΣbL, GbG))
    out[4] = -1.0im * (∫dt(t2, t1, ΣbG, GbG))

    # c-electrons
    if !isnothing(ωc)
        for i in eachindex(ωc)
            out[4+i] = -1.0im * (ωc[i] * GcL[i][t1, t2] + ∫dt(1, t1, TcG, GcL[i]) - ∫dt(1, t2, TcL, GcG[i]) - ∫dt(t2, t1, TcL, GcL[i]))
            out[4+i+length(ωc)] = -1.0im * (ωc[i] * GcG[i][t1, t2] + ∫dt(t2, t1, TcG, GcG[i]) - ∫dt(1, t1, TcL, GcG[i]) + ∫dt(1, t2, TcG, GcL[i]))
        end
    end

    return out
end

function fd!(out, model::PhotonAssistedSIAM, data::TimeData, ts, t1, t2)
    fv!(out, model, data, ts, t1, t2)
    out .-= adjoint.(out)
end

function Σ!(model::PhotonAssistedSIAM, data::TimeData, ts, t1, t2)
    @unpack GcL, GcG, GfL, GfG, GbL, GbG, GγL0, GγG0, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, ρc = data

    if (n = size(GfL, 2)) > size(ΣfL, 2)
        resize!(ΣfL, n)
        resize!(ΣfG, n)
        resize!(ΣbL, n)
        resize!(ΣbG, n)
        resize!(TcL, n)
        resize!(TcG, n)
    end

    if isnothing(ρc)
        GcL_ = GcL(ts[t1], ts[t2])
        GcG_ = GcG(ts[t1], ts[t2])
    else
        GcL_ = sum(GcL[i][t1, t2] * ρc[i] for i in eachindex(ρc))
        GcG_ = sum(GcG[i][t1, t2] * ρc[i] for i in eachindex(ρc))

        TcL[t1, t2] = +0.3im * model.V₀^2 * GfL[t1, t2] * GbG[t2, t1]
        TcG[t1, t2] = +0.3im * model.V₀^2 * GfG[t1, t2] * GbL[t2, t1]
    end

    V²L = model.V₀^2 + im * model.Γ₀^2 * (GγL0(ts[t1], ts[t2]) + GγG0(ts[t2], ts[t1]))
    V²G = model.V₀^2 + im * model.Γ₀^2 * (GγG0(ts[t1], ts[t2]) + GγL0(ts[t2], ts[t1]))

    ΣfL[t1, t2] = +1.0im * V²L * GbL[t1, t2] * GcL_
    ΣfG[t1, t2] = +1.0im * V²G * GbG[t1, t2] * GcG_
    ΣbL[t1, t2] = -2.0im * V²L * GfL[t1, t2] * (-GcG_')
    ΣbG[t1, t2] = -2.0im * V²G * GfG[t1, t2] * (-GcL_')
    return
end
