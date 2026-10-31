# Wall-E

A compact, fast UCI chess engine in C++20: bitboards with PEXT sliders, a full modern search
stack, Lazy SMP and **incremental NNUE** evaluation. It builds to one statically linked
`chess.exe` with no runtime dependencies, and the evaluation network is compiled into the
executable.

**Download:** [Wall-E v1.1.1](https://github.com/Tianl1234/Wall-e/releases/tag/v1.1.1) — a single
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

Every network that is loaded must pass a material sanity probe (pawn < minor piece < rook <
queen, with plausible magnitudes). A network that fails it is rejected, and the engine keeps the
handcrafted evaluation instead of silently playing badly.

**The bundled `fingolfin.nnue` fails that probe** — it scores a bishop as negative and a knight
below a pawn — so the shipped binary currently plays with the handcrafted evaluation. Probing
about 69k feature-layout/output interpretations of that file found none that makes it sane, so
it is a bad or mismatched network rather than a loader bug. Drop in a network trained for this
layout (768 inputs = 2 colours × 6 piece types × 64 squares) and it is picked up automatically,
from the file or from inside the executable.

## How strong is it?

There is **no official rating yet**: Wall-E has not played a rated match, so any Elo number is
an estimate, not a measurement. What was actually measured on the build machine (Ryzen 9 9900X,
single thread, shipped binary, handcrafted evaluation):

| Test | Result |
|---|---|
| bench — 6 standard positions, depth 12 | 5,454,433 nodes in 1,185 ms (**4.60 Mnps**) |
| startpos → depth 18 | 3,778 ms, 14,435,858 nodes (**3.82 Mnps**) |
| perft(6) from startpos | 119,060,324 nodes in 1.92 s (**~62 Mnps**) |
| perft positions checked (startpos, Kiwipete, position 3) | all match the reference node counts |
| incremental state self-test (`verify`) | zero Zobrist / material / PSQT / NNUE mismatches |

The numbers published for earlier releases were recorded with the unusable network loaded, so they are not
comparable with the table above; these are from v1.1.1. The engine no longer quotes an Elo
estimate: the strongest honest statement is that it plays the search stack below at a few
million nodes per second, and that a controlled match against a rated engine is the next step
before claiming a rating.

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
* the network is embedded in the executable and validated by the material sanity probe above
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
