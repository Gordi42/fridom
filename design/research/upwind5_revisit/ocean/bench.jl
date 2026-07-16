# Matched-protocol Oceananigans.jl step-time benchmark.
# Mirrors fridom's p3b_model.py build_model + timing harness.
# Usage: julia --project=ocean bench.jl SCHEME N [DT]
#   SCHEME in {centered, upwind5, weno5}; N grid size; DT optional (default 20.0)
# Fresh process per (scheme, n) run (compile/GC isolation).

using Oceananigans
using CUDA
using Printf
using JSON

const SCHEME = ARGS[1]
const N = parse(Int, ARGS[2])
const DT0 = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 20.0

advection(scheme) =
    scheme == "centered" ? Centered(order=2) :
    scheme == "upwind5"  ? UpwindBiased(order=5) :
    scheme == "weno5"    ? WENO(order=5) :
    error("unknown scheme $scheme")

function build_model(n)
    grid = RectilinearGrid(GPU();
        size=(n, n, n),
        x=(0, 10000.0), y=(0, 10000.0), z=(0, 100.0),
        topology=(Periodic, Periodic, Periodic))
    model = NonhydrostaticModel(grid;
        timestepper=:QuasiAdamsBashforth2,
        advection=advection(SCHEME),
        tracers=:b,
        buoyancy=BuoyancyTracer(),
        coriolis=FPlane(f=1e-4))
    kx = 2π/10000.0; ky = 2π/10000.0; kz = 2π/100.0
    uic(x, y, z) = 0.2 * sin(kx*x) * cos(ky*y)
    vic(x, y, z) = 0.06 * cos(kx*x)
    bic(x, y, z) = 1e-4 * cos(kz*z)
    set!(model; u=uic, v=vic, b=bic)
    return model
end

function run_chunk!(model, dt, steps)
    for _ in 1:steps
        time_step!(model, dt)
    end
    CUDA.synchronize()
end

function bench(dt)
    model = build_model(N)
    CUDA.synchronize()
    steps = 50
    # warmup chunk (compile + warm), discarded
    t0 = time_ns()
    run_chunk!(model, dt, steps)
    first_s = (time_ns() - t0) / 1e9
    # timed chunks
    ms_per_step = Float64[]
    for _ in 1:6
        t0 = time_ns()
        run_chunk!(model, dt, steps)
        push!(ms_per_step, (time_ns() - t0) / 1e6 / steps)
    end
    # finiteness check on u
    ufield = Array(interior(model.velocities.u))
    finite = all(isfinite, ufield)
    return (; first_s, ms_per_step, finite, model)
end

r = bench(DT0)
dt_used = DT0
if !r.finite
    @warn "non-finite at dt=$DT0; retrying at dt=5.0"
    r = bench(5.0)
    dt_used = 5.0
end

sorted = sort(r.ms_per_step)
med = (sorted[3] + sorted[4]) / 2   # median of 6
mn = sorted[1]

result = Dict(
    "scheme" => SCHEME,
    "n" => N,
    "dt" => dt_used,
    "first_chunk_compile_s" => r.first_s,
    "ms_per_step_median" => med,
    "ms_per_step_min" => mn,
    "ms_per_step_all" => r.ms_per_step,
    "finite" => r.finite,
    "oceananigans_version" => "0.105.3",
)

outdir = joinpath(@__DIR__, "results")
mkpath(outdir)
label = get(ENV, "BENCH_LABEL", "")
suffix = isempty(label) ? "" : "_$(label)"
outpath = joinpath(outdir, "ocean_$(SCHEME)_n$(N)$(suffix).json")
open(outpath, "w") do io
    JSON.print(io, result, 2)
end

@printf("[%s n%d dt=%.1f] ms/step median=%.3f min=%.3f finite=%s compile=%.1fs\n",
        SCHEME, N, dt_used, med, mn, r.finite, r.first_s)
println("wrote $outpath")
