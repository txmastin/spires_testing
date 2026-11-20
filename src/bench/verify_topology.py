#!/usr/bin/env python
"""Structural verification of SPIRES's three topology generators.

Exports the weight matrix SPIRES actually builds for each connectivity type
and checks it against the defining signature of that topology, using
NetworkX's own generators as an independently-implemented control:

  SPIRES_CONN_RANDOM       Erdos-Renyi: density == connectivity, Poisson
                           degree distribution, low clustering.
  SPIRES_CONN_SMALL_WORLD  Watts-Strogatz: near-uniform out-degree, high
                           clustering relative to a random graph of equal
                           density, short characteristic path length.
  SPIRES_CONN_SCALE_FREE   Barabasi-Albert: heavy-tailed degree distribution
                           on the undirected projection, hub structure.

Visual inspection alone cannot distinguish these reliably -- a spring layout
of any sparse graph looks like a hairball -- so every panel is paired with the
statistic that actually defines the family.
"""
import argparse, json, pathlib, struct, subprocess, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPORT = ROOT / "src" / "bench" / "spires_export"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"

TOPOS = [("random", "Erdos-Renyi"),
         ("small_world", "Watts-Strogatz"),
         ("scale_free", "Barabasi-Albert")]


def spires_W(n, conn, topology, seed=4, ei=0.8, sr=1.0):
    """Export the recurrent matrix SPIRES generates for this configuration."""
    out = f"/tmp/topo_{topology}_{n}_{conn}_{seed}.bin"
    subprocess.run([str(EXPORT), "--n", str(n), "--c", "1", "--conn", str(conn),
                    "--ei", str(ei), "--sr", str(sr), "--seed", str(seed),
                    "--topology", topology, "--out", out],
                   cwd=ROOT, check=True, capture_output=True)
    with open(out, "rb") as f:
        nn, cc = np.frombuffer(f.read(8), dtype="<i4")
        W = np.frombuffer(f.read(nn * nn * 8), dtype="<f8").reshape(nn, nn).copy()
    return W


def reference_graph(topology, n, conn, seed=4):
    """NetworkX's own generator for the same nominal parameters."""
    if topology == "random":
        return nx.gnp_random_graph(n, conn, seed=seed, directed=True).to_undirected()
    if topology == "small_world":
        # SPIRES lays K forward edges per node on a directed ring, so the
        # undirected projection is a WS lattice of degree 2K.
        k = 2 * max(1, int(round(conn * (n - 1))))
        if k % 2:
            k += 1
        k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
        return nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    m = max(1, int(round(conn * (n - 1))))
    m = min(m, n - 1)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def metrics(G, label):
    """Structural statistics on an undirected graph."""
    deg = np.array([d for _, d in G.degree()], dtype=float)
    n = G.number_of_nodes()
    out = dict(label=label, n=n, edges=G.number_of_edges(),
               density=nx.density(G),
               mean_degree=float(deg.mean()), std_degree=float(deg.std()),
               max_degree=float(deg.max()),
               degree_cv=float(deg.std() / max(deg.mean(), 1e-12)),
               clustering=float(nx.average_clustering(G)))
    # Path length on the largest connected component, sampled for tractability
    if G.number_of_nodes():
        cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(cc)
        rng = np.random.default_rng(0)
        nodes = list(H.nodes())
        sample = rng.choice(nodes, size=min(40, len(nodes)), replace=False)
        tot, cnt = 0, 0
        for src in sample:
            for _, d in nx.single_source_shortest_path_length(H, src).items():
                tot += d; cnt += 1
        out["avg_path_len"] = tot / max(cnt, 1)
        out["largest_cc_frac"] = len(cc) / G.number_of_nodes()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stats", type=int, default=600,
                    help="size for degree statistics and metrics")
    ap.add_argument("--n-spy", type=int, default=200,
                    help="size for the adjacency plot")
    ap.add_argument("--n-draw", type=int, default=60,
                    help="size for the node-link drawing")
    ap.add_argument("--conn", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10,
                    help="independent networks per topology for the metric statistics")
    ap.add_argument("--out", default="figures/topology_verify.pdf")
    ap.add_argument("--json", default="results/topology_verify.json")
    a = ap.parse_args()

    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0))
    fig.patch.set_facecolor("white")

    for r, (topo, nice) in enumerate(TOPOS):
        # ---------- adjacency ----------
        Wspy = spires_W(a.n_spy, a.conn, topo, a.seed)
        ax = axes[r][0]
        ax.imshow(Wspy != 0, cmap="Greys", interpolation="nearest", aspect="equal")
        ax.set_title(f"{nice}\nadjacency  (N={a.n_spy})", fontsize=10,
                     color=INK, loc="left")
        ax.set_xlabel("post-synaptic", fontsize=8, color=MUTED)
        ax.set_ylabel("pre-synaptic", fontsize=8, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)

        # ---------- degree distribution vs reference ----------
        # Metrics are computed over `--seeds` independent networks so the
        # comparison against NetworkX carries a variance, not a single draw.
        # The histogram pools the same draws; the adjacency and node-link
        # panels necessarily show one representative network.
        per_seed_s, per_seed_r, ds_all, dr_all = [], [], [], []
        dens = []
        for k in range(a.seeds):
            sd_ = a.seed + k
            Wk = spires_W(a.n_stats, a.conn, topo, sd_)
            Gk = nx.from_numpy_array((Wk != 0).astype(int),
                                     create_using=nx.DiGraph).to_undirected()
            Gk_r = reference_graph(topo, a.n_stats, a.conn, sd_)
            per_seed_s.append(metrics(Gk, f"SPIRES {nice}"))
            per_seed_r.append(metrics(Gk_r, f"NetworkX {nice}"))
            ds_all.append(np.array([d for _, d in Gk.degree()], dtype=float))
            dr_all.append(np.array([d for _, d in Gk_r.degree()], dtype=float))
            dens.append(float((Wk != 0).sum()) / (a.n_stats * (a.n_stats - 1)))
            if k == 0:
                W, Gs_u, Gr = Wk, Gk, Gk_r

        ds = np.concatenate(ds_all)
        dr = np.concatenate(dr_all)

        ax = axes[r][1]
        hi = int(max(ds.max(), dr.max()))
        bins = np.arange(0, hi + 2) - 0.5
        ax.hist(ds, bins=bins, color=BLUE, alpha=0.75, label="SPIRES", zorder=3)
        ax.hist(dr, bins=bins, histtype="step", color=ORANGE, linewidth=1.8,
                label="NetworkX reference", zorder=4)
        ax.set_title("degree distribution (undirected projection)",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("degree", fontsize=8, color=MUTED)
        ax.set_ylabel("neurons", fontsize=8, color=MUTED)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7)
        if topo == "scale_free":
            ax.set_yscale("log")

        # ---------- node-link drawing ----------
        Wd = spires_W(a.n_draw, a.conn, topo, a.seed)
        Gd = nx.from_numpy_array((Wd != 0).astype(int),
                                 create_using=nx.DiGraph).to_undirected()
        ax = axes[r][2]
        if topo == "small_world":
            pos = nx.circular_layout(Gd)     # ring structure is the point
        else:
            pos = nx.spring_layout(Gd, seed=1, k=0.6)
        degd = np.array([d for _, d in Gd.degree()], dtype=float)
        sizes = 12 + 90 * (degd / max(degd.max(), 1))
        nx.draw_networkx_edges(Gd, pos, ax=ax, edge_color=GRID, width=0.6, alpha=0.9)
        nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=sizes, node_color=BLUE,
                               linewidths=0)
        lay = "circular layout" if topo == "small_world" else "spring layout"
        ax.set_title(f"structure  (N={a.n_draw}, {lay})", fontsize=10,
                     color=INK, loc="left")
        ax.axis("off")

        def summarise(per_seed):
            keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
            out = {}
            for k in keys:
                v = np.array([d[k] for d in per_seed], dtype=float)
                out[k] = float(v.mean())
                out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out["label"] = per_seed[0].get("label", "")
            return out

        ms, mr = summarise(per_seed_s), summarise(per_seed_r)
        ms["requested_connectivity"] = a.conn
        ms["directed_density"] = float(np.mean(dens))
        ms["directed_density_sd"] = float(np.std(dens, ddof=1)) if len(dens) > 1 else 0.0
        ms["seeds"] = a.seeds
        rows.append(dict(topology=topo, spires=ms, reference=mr))

    fig.text(0.005, 0.008,
             f"SPIRES connectivity={a.conn}. Degree panels pool {a.seeds} "
             f"independent networks per generator, compared against NetworkX at "
             f"the same nominal parameters. Adjacency and structure panels show "
             f"one representative network. Scale-free y-axis is log.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(ROOT / a.out, facecolor="white")
    print("wrote", ROOT / a.out)

    (ROOT / a.json).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.json).write_text(json.dumps(rows, indent=2))

    print(f"\n{'topology':<16} {'source':<10} {'density':>8} {'meandeg':>8} "
          f"{'deg CV':>7} {'clust':>7} {'path':>6} {'maxdeg':>7}")
    for row in rows:
        for src in ("spires", "reference"):
            m = row[src]
            print(f"{row['topology']:<16} {src:<10} {m['density']:>8.4f} "
                  f"{m['mean_degree']:>8.2f} {m['degree_cv']:>7.3f} "
                  f"{m['clustering']:>7.4f} {m.get('avg_path_len', 0):>6.2f} "
                  f"{m['max_degree']:>7.0f}")
    print(f"\nrequested connectivity = {a.conn}")
    for row in rows:
        d = row["spires"]["directed_density"]
        print(f"  {row['topology']:<16} directed density {d:.4f}  "
              f"(ratio to requested {d/a.conn:.3f})")


if __name__ == "__main__":
    main()
