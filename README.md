# Wall-E

A compact, fast UCI chess engine in C++20: bitboards with PEXT sliders, a full modern search
stack, Lazy SMP and **incremental NNUE** evaluation. It builds to one statically linked
`chess.exe` with no runtime dependencies, and the evaluation network is compiled into the
executable.

**Download:** [Wall-E v1.4.0](https://github.com/Tianl1234/Wall-e/releases/tag/v1.4.0) — a single
self-contained `chess.exe` with the NNUE network embedded. The engine reports itself as
`Wall-E 2` over UCI.

```
Wall-e/
  chess.exe            <- release binary (Windows x64, static, network embedded)
  chess.cpp            <- engine: search, move generation, UCI
  nnue.cpp / nnue.h    <- incremental NNUE evaluation + handcrafted fallback
  nnue_net.cpp         <- embeds nnue/fingolfin.nnue into chess.exe
  build.sh             <- build script (w64devkit GCC)
  nnue/
    fingolfin.nnue     <- evaluation network (compiled into chess.exe)
    PATRICIA-LICENSE.txt
```

At startup the engine looks for `nnue/fingolfin.nnue` in the working directory, then next to the
executable, and finally falls back to the network compiled into the binary — so one file is
enough to run it.

## Evaluation sanity check

Every network that is loaded must pass a material sanity probe before it is used, so a corrupt
or mismatched file cannot silently ruin the evaluation and make a GUI's evaluation bar
meaningless. The probe takes the **start position** and removes one white piece at a time
(pawn, knight, bishop, rook, queen): the resulting deltas have to form a sane material scale.

It deliberately avoids bare-kings positions. A lone minor piece, or a pawn that cannot promote,
is a dead draw, so a correctly trained network rates it near zero — an ordering test there
rejects perfectly good networks (it did, in v1.1.0/v1.1.1).

The bundled `fingolfin.nnue` passes comfortably. Measured with this build, removing a white
piece from the start position costs: pawn 49 cp, knight 483, bishop 500, rook 641, queen 1033 —
a normal trained-network scale. A network that does fail the probe is rejected and the engine
keeps the handcrafted evaluation instead. A network trained for this layout (768 inputs =
2 colours × 6 piece types × 64 squares) is picked up automatically, from the file or from inside
the executable.

## How strong is it?

There is **no official rating yet**: Wall-E has not played a rated match, so any Elo number is
an estimate, not a measurement. What was actually measured on the build machine (Ryzen 9 9900X,
single thread, shipped binary, NNUE evaluation active):

| Test | Result |
|---|---|
| bench — 6 standard positions, depth 12 | 797,013 nodes in 309 ms (**2.58 Mnps**) |
| bench — depth 16 | 6,929,453 nodes in 2.72 s (**2.54 Mnps**) |
| startpos → depth 18 | 1,131 ms, 3,224,082 nodes (**2.85 Mnps**) |
| perft(6) from startpos | 119,060,324 nodes in 1.82 s (**~65 Mnps**) |
| perft positions checked (startpos, Kiwipete, position 3) | all match the reference node counts |
| incremental state self-test (`verify`) | zero Zobrist / material / PSQT / NNUE mismatches |

These figures are from v1.4.0. The v1.1.0/v1.1.1 numbers were taken with the network rejected
and are not comparable. The engine does not quote an Elo estimate: a controlled match against a
rated engine is the next step before claiming a rating.

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
* the network is embedded in the executable and checked by the material sanity probe above
  before it is used
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

The release binary is built with the [w64devkit](https://github.com/skeeto/w64devkit) GCC.
`nnue/fingolfin.nnue` must exist at build time: `nnue_net.cpp` pulls it into the executable's
read-only data with `.incbin`, so the result is a single self-contained binary.

```sh
g++ -O3 -march=znver5 -std=c++20 -DNDEBUG -s \
    -static -static-libgcc -static-libstdc++ \
    -o chess.exe chess.cpp nnue.cpp nnue_net.cpp
```

`build.sh` does exactly this (from the project root, which `.incbin` requires) and is the
recommended entry point. Static linking is required for GUIs: a dynamically linked build needs
`libstdc++-6.dll` / `libgcc_s_seh-1.dll`, which are not on the system PATH, and the engine then
dies at process start (seen as a disconnect in En Croissant and Arena).

The stripped executable is about 2.9 MiB, of which 1.58 MB is the embedded network; without it
the binary is about 1.3 MiB.

`-march=znver5` targets Zen 4/5 (AVX-512, BMI2). On older CPUs rebuild with
`-march=x86-64-v3` (AVX2, BMI2) or `-march=x86-64` (portable, much slower NNUE path) by
setting the equivalent flags in `build.sh`. The PEXT bitboard backend needs BMI2; on CPUs
without it, replace `_pext_u64` with the magic-bitboard path.

## Using it in a GUI

Add `chess.exe` as a UCI engine in En Croissant, Arena, Cute Chess or BanksiaGUI. Nothing else
is needed — the network travels inside the executable.

## Changes in v1.4.0

Review pass over the desktop engine. Fixes:

* **TT replacement could overwrite the wrong entry** — the victim-selection loop compared each
  candidate against a slot that was reassigned inside the loop, so fresh, deep entries could be
  dropped while old, shallow ones survived
* **stale root continuation history** — ply 1 read the counter-move and continuation-history key
  from an old root move; the root move is now stored before the child search
* **Lazy SMP helpers searched at a fixed depth** instead of climbing with the main thread; they
  now run the same iterative-deepening loop, which noticeably improves multi-thread depth
* **a failed NNUE reload removed the working evaluator** — a bad `setoption name EvalFile` left
  the engine with no network; the previous network is now kept until the replacement passes the
  sanity probe
* **movetime left 40% of the budget unused** — the soft limit is now 90% of the allotted time
  (still inside the hard limit)

Speed:

* SEE is iterative instead of recursive (used by move ordering and SEE pruning)
* transposition-table bucket prefetch at node entry
* the shared node counter is only updated atomically when a node limit is set; normal search
  uses plain per-thread counters, summed for `info nodes`

Measured against the previous build: **2.56 vs 2.51 Mnps** on the short bench, **~2.6% fewer
nodes** at equal depth on the 6-position bench, and **depth 20 vs 18** with 4 threads at 3 s.
Perft, `verify 4` (zero inconsistencies), fixed-depth move comparison, self-play and the
rejected-network test all pass on the shipped binary.

## Changes in v1.1.2

The sanity probe added in v1.1.1 was wrong, and this release corrects it. It probed bare-kings
positions holding a single piece and required a monotone material ordering there. Those
positions are outside a trained network's distribution — a lone bishop or knight cannot mate, so
the network correctly rates such positions as a draw — and the probe therefore rejected a
perfectly good network and left the engine playing with the handcrafted evaluation.

The probe now measures the start position, and the shipped binary evaluates with NNUE again:
removing a white piece from the start position costs 49 cp for the a2 pawn, 483 for the knight,
500 for the bishop, 641 for the rook and 1033 for the queen.

## Changes in v1.1.1

Bug fixes found while reviewing the engine; all of them were verified on the shipped binary:

* **repetition detection never fired** — the history scan started one ply off, so it only ever
  compared positions with the opposite side to move. Forced perpetuals were scored as losses
  instead of draws. The scan is now bounded, and the history buffer compacts instead of
  overflowing once full
* **castling** moves were generated from FEN rights alone; rights are now validated against
  king/rook placement, and `remove_piece` no longer uses XOR — which could fabricate a phantom
  rook and desynchronise the position
* `Undo` stored the halfmove clock in a `uint8_t`, silently truncating clocks above 255
* `setoption Hash` reallocated the transposition table while the search thread was using it; the
  search is now stopped first
* unusable networks are rejected by the material sanity probe (see above) instead of being used
* Lazy SMP helpers restarted iterative deepening from depth 1 on every iteration
* the network is now compiled into `chess.exe` (`nnue_net.cpp`), so a release is one file

## License and credits

Code: MIT — see [LICENSE](LICENSE).

The NNUE network `nnue/fingolfin.nnue` is Patricia's network by Adam Kulju, used under the MIT
license (see [nnue/PATRICIA-LICENSE.txt](nnue/PATRICIA-LICENSE.txt)). Patricia is not
included in this repository, only its evaluation network.
