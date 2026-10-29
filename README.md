# Wall-E

A compact, fast UCI chess engine in C++20: bitboards with PEXT sliders, a full modern search
stack, Lazy SMP and **incremental NNUE** evaluation. It builds to one statically linked
`chess.exe` with no runtime dependencies.

**Download:** [Wall-E version 1](https://github.com/Tianl1234/Wall-e/releases/tag/v1.0.0) —
`chess.exe` plus the NNUE network `fingolfin.nnue`. The engine reports itself as `Wall-E 2`
over UCI; this is the first public release.

```
Wall-e/
  chess.exe            <- release binary (Windows x64, static)
  nnue/
    fingolfin.nnue     <- evaluation network (required for full strength)
```

The engine looks for `nnue/fingolfin.nnue` in the working directory first, then next to the
executable, so both layouts work. Without the network it falls back to a handcrafted
evaluation — it still plays, but noticeably weaker.

## How strong is it?

There is **no official rating yet**: Wall-E has not played a rated match, so any Elo number is
an estimate, not a measurement. What was actually measured on the build machine
(Ryzen 9 9900X, single thread, shipped binary, `bench 12`):

| Test | Result |
|---|---|
| bench — 6 standard positions, depth 12 | 912,862 nodes in 390–395 ms (**2.31–2.34 Mnps**) |
| startpos → depth 18 | 1,105 ms, 2,993,729 nodes (**2.71 Mnps**) |
| perft(6) from startpos | 119,060,324 nodes in 1.97 s (**~60 Mnps**) |
| six standard perft positions | all match (startpos, Kiwipete, positions 3–6) |
| incremental state self-test (`verify`) | zero Zobrist / material / PSQT / NNUE mismatches |

With NNUE evaluation on top of the search stack described below, the engine is estimated to
play in the **~2400 Elo class at blitz** on modern single-threaded hardware. That is an
inference from the architecture and search depth, not a result of games against rated
opponents; a controlled match against a rated engine is the next step before quoting a number.

## Search

* principal variation search with aspiration windows, root move re-ordering between iterations
* transposition table: 4-way 16-byte buckets, aging, mate-score adjustment, depth-preferred
* pruning/reduction: null move, reverse futility, razoring, ProbCut, internal iterative
  reduction, late-move reductions adjusted by history, late-move pruning, futility pruning,
  SEE pruning, check extension, mate-distance pruning
* move ordering: TT move, winning/equal captures by SEE, promotions, killers, butterfly and
  continuation history, counter-moves
* quiescence search with SEE-based capture ordering, played from a 16-bit move encoding with
  zero allocation on the hot path
* Lazy SMP across up to 128 threads sharing one hash table

## Evaluation

* NNUE: 768 inputs (own/opponent perspective × piece × square), 1024 hidden units, one
  output neuron with a quadratic activation layer in Patricia's network format
* int16 accumulators updated **incrementally** on make/unmake, batched per move, saturating
  in SIMD (AVX-512 → AVX2 → scalar fallback)
* handcrafted fallback: PeSTO material/PSQT plus mobility, pawn structure (doubled, isolated,
  backward, supported, passed by rank), rook files and 7th rank, bishop pair, threats,
  king shelter and attack units, space, tempo — all tapered by game phase

## UCI options

| Option | Default | Range |
|---|---|---|
| `Hash` | 256 MB | 1–8192 |
| `Threads` | 1 | 1–128 |
| `Move Overhead` | 30 ms | 0–5000 |
| `EvalFile` | `nnue/fingolfin.nnue` | any path; reloads the network at runtime |

Standard protocol is supported (`uci`, `isready`, `setoption`, `ucinewgame`, `position
[startpos|fen] ... moves ...`, `go` with clocks/depth/movetime/infinite/ponder, `stop`,
`ponderhit`, `quit`). Search runs asynchronously, so `stop` and `isready` are answered during
a search. Extra console commands for testing: `d` (board + hash), `eval`, `perft N`,
`verify N` (recomputes all incremental state over a perft tree), `bench [depth]`.

## Build

The release binary is built with the [w64devkit](https://github.com/skeeto/w64devkit) GCC. `build.sh` strips debug/symbol tables (`-s`) to keep the executable small; the NNUE network remains a separate 1.58 MB file:

```sh
g++ -O3 -march=znver5 -std=c++20 -DNDEBUG -s \
    -static -static-libgcc -static-libstdc++ \
    -o chess.exe chess.cpp nnue.cpp
```

`build.sh` does exactly this and is the recommended entry point. Static linking is required
for GUIs: a dynamically linked build needs `libstdc++-6.dll` / `libgcc_s_seh-1.dll`, which are
not on the system PATH, and the engine then dies at process start (seen as a disconnect in
En Croissant and Arena).

On the Ryzen 9 9900X test system, the stripped executable is approximately 1.3 MiB; the previous unstripped binary was approximately 3.1 MiB. The NNUE incremental-update optimization removed redundant scratch-buffer work and roughly doubled the measured `bench 12` throughput. This is a node-rate result, not a measured Elo increase.

`-march=znver5` targets Zen 4/5 (AVX-512, BMI2). On older CPUs rebuild with
`-march=x86-64-v3` (AVX2, BMI2) or `-march=x86-64` (portable, much slower NNUE path) by
setting `CXXFLAGS`-equivalent flags in `build.sh`. The PEXT bitboard backend needs BMI2; on
CPUs without it, replace `_pext_u64` with the magic-bitboard path.

## Using it in a GUI

Add `chess.exe` as a UCI engine in En Croissant, Arena, Cute Chess or BanksiaGUI and keep the
`nnue/` folder beside the executable. Nothing else is needed.

## License and credits

Code: MIT — see [LICENSE](LICENSE).

The NNUE network `nnue/fingolfin.nnue` is Patricia's network by Adam Kulju, used under the MIT
license (see [nnue/PATRICIA-LICENSE.txt](nnue/PATRICIA-LICENSE.txt)). Patricia is not
included in this repository, only its evaluation network.
