#!/usr/bin/env python
"""Per-implementation build cost: what it costs to turn an LSM description into
something runnable, for each framework.

This is deliberately not the cost of installing the framework. Building
libspires.a, or pip-installing PyTorch, is paid once and amortised over every
network afterwards, so neither belongs here. What belongs here is the work that
recurs whenever the network changes:

  SPIRES    compiling the C program that describes the LSM against the
            prebuilt library
  Brian2    cpp_standalone generating and compiling a dedicated binary per
            network (measured in the sweep itself, as codegen_s)
  ANNarchy  the same, via its C++ code generator (also codegen_s)
  others    nothing: the network is described in a script the interpreter
            already runs, so there is no build step to pay

Writes results/build_costs.json. Frameworks whose cost is measured per cell in
the sweep are absent here; make_tables.py takes those from codegen_s.
"""
import json, os, pathlib, shlex, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPIRES = pathlib.Path(os.environ.get(
    "SPIRES_DIR", pathlib.Path.home() / "programming/projects/spiking-reservoir/spires"))
REPS = 6

# The library is linked, not rebuilt: -fopenmp and clang because libspires.a
# carries LLVM OpenMP symbols and will not link against libgomp.
CMD = (f"clang -O2 -fopenmp -I{SPIRES}/include -I{SPIRES}/src "
       f"{ROOT}/src/bench/run_spires.c -o {{out}} "
       f"{SPIRES}/lib/libspires.a -lopenblas -lm")


def time_compile(out):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        p = subprocess.run(shlex.split(CMD.format(out=out)),
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if p.returncode:
            sys.exit(f"compile failed:\n{p.stderr}")
        ts.append(dt)
    return ts


def main():
    out = ROOT / "results" / ".build_cost_probe"
    ts = time_compile(out)
    out.unlink(missing_ok=True)

    doc = {
        "note": "per-implementation build cost, seconds; see module docstring",
        "reps": REPS,
        "frameworks": {
            "SPIRES": {
                "build_s": min(ts),
                "all_s": ts,
                "what": "clang -O2 -fopenmp of the LSM program, linked "
                        "against the prebuilt libspires.a",
                "command": CMD.format(out="<binary>"),
            },
        },
        "zero_by_construction": [
            "reservoirpy", "pytorch-esn", "snnTorch", "norse", "rockpool",
            "NEST", "ReservoirComputing.jl",
        ],
        "measured_per_cell_as_codegen_s": ["brian2", "ANNarchy"],
    }
    dst = ROOT / "results" / "build_costs.json"
    dst.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"SPIRES LSM-program build: min {min(ts):.3f} s "
          f"over {REPS} reps -> {dst}")


if __name__ == "__main__":
    main()
