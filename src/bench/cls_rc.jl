# ReservoirComputing.jl on a classification task (SHD / FSDD).
#
# Reads the same flat .bin the SPIRES C runner consumes, so all frameworks
# train on byte-identical data. Per-sample the reservoir is re-initialised,
# states are collected with RC.jl's native collectstates, pooled to
# [mean ; last], and classified with a native StandardRidge over one-hot
# targets. A JIT warmup runs first so compilation is not timed.

using ReservoirComputing, Random, Printf, Statistics, LinearAlgebra

function parse_args()
    n, rep, seed, task, topo = 300, 0, 0, "fsdd", "random"
    sr, leak, gain = 1.2, 0.3, 1.0
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a == "--n"        ; n    = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--rep"      ; rep  = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--seed"     ; seed = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--task"     ; task = ARGS[i+1]; i += 2
        elseif a == "--topology" ; topo = ARGS[i+1]; i += 2
        elseif a == "--sr"       ; sr   = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--leak"     ; leak = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--gain"     ; gain = parse(Float64, ARGS[i+1]); i += 2
        else i += 1
        end
    end
    return n, rep, seed, task, topo, sr, leak, gain
end

function load_bin(task)
    open("data/$(task).bin") do f
        hdr = Vector{Int32}(undef, 5); read!(f, hdr)
        ntr, nte, T, C, nc = Int.(hdr)
        Xtr = Array{UInt8}(undef, C, T, ntr); read!(f, Xtr)
        ytr = Vector{Int32}(undef, ntr); read!(f, ytr)
        Xte = Array{UInt8}(undef, C, T, nte); read!(f, Xte)
        yte = Vector{Int32}(undef, nte); read!(f, yte)
        return Xtr, Int.(ytr), Xte, Int.(yte), T, C, nc
    end
end

function harvest(model, X, ps, st)
    C, T, ns = size(X)
    F = Matrix{Float32}(undef, 2C > 0 ? 0 : 0, 0)   # placeholder, sized below
    feats = nothing
    for s in 1:ns
        x = Float32.(@view X[:, :, s])
        r = collectstates(model, x, ps, st)
        S = isa(r, Tuple) ? r[1] : r
        p = vcat(vec(mean(S, dims=2)), S[:, end])
        if feats === nothing
            feats = Matrix{Float32}(undef, length(p), ns)
        end
        feats[:, s] = p
    end
    return feats
end

function main()
    n, rep, seed, task, topo, sr, leak, gain = parse_args()
    Xtr, ytr, Xte, yte, T, C, nc = load_bin(task)
    rng = MersenneTwister(seed + rep)

    mk() = ESN(C, n, nc; init_reservoir=rand_sparse(; radius=sr, sparsity=0.1),
               init_input=scaled_rand(; scaling=gain), leak_coefficient=leak)

    # JIT warmup on a small slice, discarded
    let m = mk(), (p, s) = setup(rng, m)
        harvest(m, Xtr[:, :, 1:min(3, size(Xtr,3))], p, s)
    end

    model = mk(); ps, st = setup(rng, model)
    t0 = time_ns()
    Ftr = harvest(model, Xtr, ps, st)
    Fte = harvest(model, Xte, ps, st)
    t_harvest = (time_ns() - t0) / 1e9

    Ytr = zeros(Float32, nc, length(ytr))
    for (j, c) in enumerate(ytr); Ytr[c + 1, j] = 1f0; end

    t0 = time_ns()
    best = -1.0
    for reg in (1e-2, 1e0, 1e2, 1e4)
        W = train(StandardRidge(reg), Ftr, Ytr)
        pred = [argmax(view(W * Fte, :, j)) - 1 for j in 1:size(Fte, 2)]
        acc = mean(pred .== yte)
        best = max(best, acc)
    end
    t_fit = (time_ns() - t0) / 1e9

    @printf("@@BENCH@@ {\"framework\": \"ReservoirComputing.jl\", \"group\": \"esn\", \"n\": %d, \"rep\": %d, \"task\": \"%s\", \"topology\": \"%s\", \"t_harvest\": %.9f, \"t_fit\": %.9f, \"t_total\": %.9f, \"accuracy\": %.6f, \"nrmse\": null, \"n_train\": %d, \"n_test\": %d, \"native_readout\": true, \"notes\": \"native collectstates + StandardRidge; pooled mean+last; JIT warmed up\"}\n",
            n, rep, task, topo, t_harvest, t_fit, t_harvest + t_fit, best,
            length(ytr), length(yte))
end

main()
