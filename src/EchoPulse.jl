using LinearAlgebra
using SpecialFunctions

using NLsolve

using KadanoffBaym, FFTW, Interpolations
using KadanoffBaym: ft

using LangrethsRules

include("utils.jl")
include("model.jl")
include("nca.jl")
