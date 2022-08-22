using LinearAlgebra
using SpecialFunctions

using NLsolve

using KadanoffBaym
using LangrethsRules

using FFTW
using KadanoffBaym: ft

using Interpolations
using DataInterpolations: QuadraticInterpolation

using Test

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

include("utils.jl")
include("model.jl")
include("eom.jl")
include("nca.jl")
