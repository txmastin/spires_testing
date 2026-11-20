# ReservoirComputing.jl runner.
#
# The standard tanh ESN, using RC.jl's native rand_sparse init and its native
# StandardRidge readout, i.e. idiomatic usage.
#
# RC.jl also ships LIFESN, which is NOT spiking: there "LIF" abbreviates
# *Local Information Flow* (Liu 2025), not Leaky Integrate-and-Fire. It is a
# tanh ESNCell wrapper and was dropped from this benchmark as out of scope.
# ReservoirComputing.jl therefore contributes no spiking model here.
#
# Julia JIT-compiles on first call, so a warmup pass at small N runs before
# any timing: without it the first measurement is dominated by compilation.

using ReservoirComputing, Random, DelimitedFiles, Printf

function parse_args()
    n, rep, seed, model, topo, task = 200, 0, 0, "esn", "random", "narma10"
    sr, leak, gain, ridge = 1.2, 1.0, 0.1, 1e-6
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a == "--n"        ; n = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--rep"      ; rep = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed"     ; seed = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--model"    ; model = ARGS[i+1]; i += 2
        elseif a == "--topology" ; topo = ARGS[i+1]; i += 2
        elseif a == "--task"     ; task = ARGS[i+1]; i += 2
        elseif a == "--sr"       ; sr = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--leak"     ; leak = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--gain"     ; gain = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--ridge"    ; ridge = parse(Float64, ARGS[i+1]); i += 2
        else i += 1
        end
    end
    return n, rep, seed, model, topo, task, sr, leak, gain, ridge
end

build(model, n, sr, leak, gain) = model == "esn" ?
    ESN(1, n, 1; init_reservoir=rand_sparse(; radius=sr, sparsity=0.1),
        init_input=scaled_rand(; scaling=gain), leak_coefficient=leak) :
    error("unknown model $model")


# Process CPU time over the same region as the wall clock. CLOCK_PROCESS_CPUTIME_ID
# is 2 on Linux. For a single-threaded run this tracks wall time; a shortfall
# means the process was descheduled and the measurement saw competing load.
function cpu_now()
    ts = Ref{NTuple{2, Int64}}((0, 0))
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), 2, ts)
    t = ts[]
    return t[1] + t[2] * 1e-9
end

# `fit_cols`: the columns (time steps) the readout is fit on. The reservoir is
# still run over all of X; only the ridge sees the training segment.
function run_once(model, n, rng, X, Y, sr, leak, gain, ridge, fit_cols; timeit=true)
    m = build(model, n, sr, leak, gain)
    ps, st = setup(rng, m)
    t0 = time_ns()
    c0 = cpu_now()
    r = collectstates(m, X, ps, st)
    S = isa(r, Tuple) ? r[1] : r
    t_harvest = (time_ns() - t0) / 1e9
    cpu_harvest = cpu_now() - c0
    t0 = time_ns()
    c0 = cpu_now()
    wout = train(StandardRidge(ridge), S[:, fit_cols], Y[:, fit_cols])
    t_fit = (time_ns() - t0) / 1e9
    cpu_fit = cpu_now() - c0
    return t_harvest, t_fit, cpu_harvest, cpu_fit, S, wout
end

function main()
    n, rep, seed, model, topo, task, sr, leak, gain, ridge = parse_args()
    data = readdlm("data/$(task).csv", ',', Float64; skipstart=1)
    u = data[:, 1]; y = data[:, 2]
    # Held-out split, mirroring common.py (0-based Python rows
    # [wash, TRAIN_END) are 1-based columns wash+1:TRAIN_END).
    wash, TRAIN_END, VAL_END = 200, 2000, 3000
    X = reshape(Float32.(u), 1, :)
    Y = reshape(Float32.(y), 1, :)
    rng = MersenneTwister(seed + rep)

    # JIT warmup at small N -- discarded
    run_once(model, 32, MersenneTwister(0), X[:, 1:200], Y[:, 1:200], sr, leak, gain, ridge, 1:200)

    t_harvest, t_fit, cpu_harvest, cpu_fit, S, wout =
        run_once(model, n, rng, X, Y, sr, leak, gain, ridge, wash+1:TRAIN_END)

    pred = vec(Float64.(wout * S))
    nrm(r) = sqrt(sum(abs2, pred[r] .- y[r]) / length(r)) / std(y[r], corrected=false)
    nr_train = nrm(wash+1:TRAIN_END)
    nr_val   = nrm(TRAIN_END+1:VAL_END)
    nr       = nrm(VAL_END+1:length(y))

    name = "ReservoirComputing.jl"
    grp  = "esn"
    @printf("@@BENCH@@ {\"framework\": \"%s\", \"group\": \"%s\", \"n\": %d, \"rep\": %d, \"topology\": \"%s\", \"task\": \"%s\", \"t_harvest\": %.9f, \"t_fit\": %.9f, \"cpu_harvest\": %.9f, \"cpu_fit\": %.9f, \"cpu_total\": %.9f, \"t_total\": %.9f, \"nrmse\": %.6f, \"nrmse_val\": %.6f, \"nrmse_train\": %.6f, \"split\": \"fit[%d:%d) val[%d:%d) test[%d:%d)\", \"native_readout\": true, \"notes\": \"native rand_sparse init + StandardRidge; JIT warmed up; rate-coded\"}\n",
            name, grp, n, rep, topo, task, t_harvest, t_fit, cpu_harvest, cpu_fit, cpu_harvest + cpu_fit, t_harvest + t_fit, nr, nr_val, nr_train,
            wash, TRAIN_END, TRAIN_END, VAL_END, VAL_END, length(y))
end

using Statistics
main()
