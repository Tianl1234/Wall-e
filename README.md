# wall-e android version

This folder is a **separate copy** of the Wall-E chess engine, ported to AArch64 and
packaged as an Android engine APK. The original desktop engine in
`../Chess-engine/Chess-engine` is untouched; nothing here writes back to it.

The APK has **no user interface at all**: no activity, no chessboard, no console, no
launcher icon. It is a headless UCI engine in the format Android chess GUIs discover
(Open Exchange), so it is installed like an engine and appears only in a GUI's engine
list. `android/README.md` documents the protocol and how to import it.

## Download

- [wall-e.tiny.apk](https://github.com/Tianl1234/Wall-e/releases/download/v1.3.7-android/wall-e.tiny.apk)
  — one network (fingolfin), 2.2 MB.
- [wall-e.strong.apk](https://github.com/Tianl1234/Wall-e/releases/download/v1.3.7-android/wall-e.strong.apk)
  — three phase networks (fingolfin/finarfin/feanor), 5.9 MB.

Both are versionCode 12 / versionName 1.3.7, signed with the same key as earlier
releases and upgrade in place. The same files are kept under versioned names in `apk/`
(`Wall-E-engine-1.3.7.apk` and `Wall-E-engine-1.3.7-strong.apk`).

## Layout

```
chess.cpp              engine sources (ported: portable sliders, guarded x86 intrinsics)
nnue.cpp / nnue.h      NNUE evaluation (ported: NEON paths, memory loader)
nnue_embedded.cpp      appends the network to the binary via .incbin (Android builds)
nnue/fingolfin.nnue    network, MIT, (c) Adam Kulju — see nnue/PATRICIA-LICENSE.txt
build.sh               desktop (x86-64) build, unchanged
android/               engine package: manifest, provider, enginelist, build script
android/README.md      format details, import steps, verification
build/                 local build output (host test binaries, APK staging)
```

## Build the APK

```sh
cd android
VARIANT=tiny   bash build_apk.sh   # -> ../build/wall-e.tiny.apk   (one network)
VARIANT=strong bash build_apk.sh   # -> ../build/wall-e.strong.apk (three phase networks)
```

Requires JDK 17+ and an Android SDK with `platforms;android-34`,
`build-tools;36.1.0` (D8 in 34.0.0 crashes on modern javac output) and
`ndk;27.3.13750724`. No Gradle.

## Engine changes for ARM and version 1.3

The desktop engine hard-coded BMI2 (`_pext_u64`, `_pdep_u64`) and x86 SIMD headers, so it
could not compile for a phone. Three changes fix that without touching the search:

1. **Portable slider backend.** Magic bitboards with multipliers generated offline
   (`build/gen_magics.cpp`) and embedded as constants. Chosen over a runtime search because
   the hardest squares need hundreds of thousands of attempts, which is far too slow at
   app launch. A collision check at table build time turns a wrong constant into a loud
   failure rather than silent misbehaviour.
   The x86 build still uses PEXT; `-DWALLE_NO_PEXT` forces the portable path for testing.
2. **NEON NNUE.** `nnue.cpp` gained NEON implementations of the three hot kernels
   (saturating accumulator add, batched delta apply, output-layer inference), written to
   match the x86 arithmetic exactly: wrapping int16 products, 32-bit accumulation, same
   rounding. Scalar code remains as the final fallback.
3. **Embedded network.** A GUI executes the engine from the APK's lib directory, where no
   `nnue/` folder exists, so Android builds link the network into the binary
   (`nnue_embedded.cpp`) and load it with `Nnue::load_embedded()`.
4. **Incremental NNUE optimization (v1.3).** Apply each added/removed feature directly to
   each perspective's int16 accumulator with saturating SIMD operations (NEON on ARM64),
   rather than allocating and reducing a two-perspective int32 delta buffer on each move.
5. **PIE.** Android requires position-independent executables; the NDK build produces one.

## Changes in 1.3.7

This release fixes the reason the evaluation bar travelled on its own while a game was
being reviewed, and it applies to the tiny package as well as the strong one.

- **The networks carry an oversized side-to-move term.** On the symmetric start position
  the nets return about +1.1 pawns for whichever side is to move — fingolfin 112 cp,
  finarfin 83, feanor 121, the same number from either perspective. Real chess puts
  roughly +0.2 there. Negamax flips the sign of the score at every ply, so a position's
  value depended on the parity of the search: it scored about a pawn higher when the net
  was the side to move, which is what made the bar move between consecutive depths and
  between consecutive reviewed moves while the position stood still.
- **The excess is measured at load and subtracted.** `nnue.cpp` now probes each network on
  the start board, where the score *is* that bonus, and stores half of it as
  `Network::tempoCorrection` (56 cp for fingolfin, 41 for finarfin, 60 for feanor).
  `evaluate()` returns `infer(...) - tempoCorrection`. The inference kernels, the feature
  indexing and the network files are unchanged, and every network carries its own
  constant, so phase-network switching stays correct.
- **Measured effect** (start position, `Threads=2`, depths 1–16, white-relative). Before,
  the score went −5, 112, 26, 103, 39, 119, 46, 92, 34, 88, 26, 92 — steps of up to
  109 cp, with 8–11 of the 15 depth changes at or above 25 cp. It now reads 51, 59, 47, 49,
  27, 46, 51, 65, 60, 61, 67, 80, 84, 71, 81: the largest step is 22 cp and at most two of
  the 15 reach 25 cp. Reviewing a 40-move game at depth 12 moves the score by 24 cp on
  average, and the remaining large steps line up with captures rather than with depth
  parity.
- **This is not a strength claim.** The network's evaluation is unchanged apart from the
  constant; the start position simply reads 56 cp instead of 112, which is a normal
  opening score. Because eval-dependent pruning now sees the lower number, `bench 12`
  visits 1,153,781 nodes instead of 556,713 at the same speed (3.4 Mnps). No match was
  played to measure the Elo effect.
- **Checked on a host build from the same sources.** `perft 5` = 4,865,609, `verify 4` =
  197,281 nodes with 0 inconsistent, a 40-move replay of the incremental accumulator
  shows 0 inconsistencies, and the evaluation is identical between the AVX-512 build and
  an SSE4.2 build with the runtime AVX2 dispatch that the x86_64 package uses.

## Changes in 1.3.6

This release removes the main reason the evaluation bar wandered while the strong package
was analysing an opening position.

- **Phase-network selection now matches desktop 1.5.0.** The strong package evaluates
  with fingolfin everywhere and switches to finarfin only when at most half the material
  is left (`phaseInc <= 12` of 24). Previously it selected feanor from `phaseInc >= 19`
  upwards — that is almost every opening and middlegame position — and changed networks a
  second time at 8.
- **Why that mattered.** Feanor is trained for speculative attacks. Over 12 opening lines
  its scores were systematically higher and moved more from depth to depth: 7 sign flips
  and a 47.9 cp average maximum step from depth 8, against 5 and 34.7 cp for fingolfin
  (finarfin was calmest, 0 and 20.8 cp). A GUI redraws the bar at every `info` line, so
  the noisier net made the bar travel further between consecutive depths. Opening
  analysis now uses the net that measured calmest among the general-purpose choices.
- **Feanor is still embedded** in the strong package, but it is no longer chosen
  automatically; a search that explicitly loads it with `EvalFile` still uses it. The
  tiny package is unchanged: it only ever contained fingolfin.
- **Checks on a host build from the same sources.** Start position and a 16-material-point
  middlegame auto-select fingolfin and produce exactly the pinned-fingolfin scores; a
  12-point endgame auto-selects finarfin and matches pinned-finarfin. `verify 4` =
  197,281 nodes / 0 inconsistent, `bench 12` = 556,713 nodes, `perft 5` = 4,865,609.

The bar still follows the score as the search deepens, and a changing best move can move
it; what this removes is the extra, network-induced swing during opening analysis.

## Changes in 1.3.5

This release cuts per-node evaluation cost, trims qsearch work and gives the phone
package a second search thread by default.

- **Batched accumulator updates.** A move changes at most two features per perspective
  (castle, capture-promotion). `make_move` now collects all adds and subs into one
  `Nnue::Update` and applies them in a single load/update/store pass, adds before subs,
  instead of one full accumulator pass per changed feature. Mirrors Patricia's update
  kernel (MIT, Adam Kulju), whose networks this engine embeds.
- **Per-ply accumulator stack.** The search keeps one accumulator per ply in
  `ThreadData::nnueStack`, computed forwards on `make_move`; `unmake` only pops. The old
  design applied the inverse update on every unmake, so taking a move back cost as much
  as making it. Updates are materialized lazily from the nearest computed ancestor, with
  a prefetch of the next update's feature rows and a per-slot evaluation cache. `verify`
  walks the same stack path and still reports zero inconsistencies over millions of
  nodes.
- **SCReLU flatten with eight accumulators.** The output layer (`infer_avx2`, NEON
  `infer`) now keeps eight independent int32 accumulators. Products are `a · (a · w)`
  in wrapping int16 then widened, exactly as before, so evaluation is bit-identical;
  the extra chains just hide multiply latency.
- **Transposition table in qsearch.** Quiet and capture cutoffs now consult the shared
  table before searching captures (Stockfish construct), and a stored move is tried
  first when it is a capture or promotion. On the bench set this removes about 16% of
  the nodes at the same depth.
- **Two threads by default on Android.** The Android package starts with `Threads=2`,
  since phone GUIs rarely change it and the phone has several cores. A 16-game host
  match at 100 ms per move scored 10-4 (2 draws) for two threads against one; desktop
  test builds keep the single-thread default. The table is shared between threads.
- **Phase-network consistency (strong package).** Choosing a different network for a
  search now clears the transposition table first, so scores produced under one network
  are not reused as if they came from the network that is now active.
- **Time management.** `go movetime X` searches until 90% of `X - Move Overhead`
  (was 60%), so a GUI that grants 2 s no longer gets a ~1 s move.
- **Transposition table replacement.** On a bucket miss the store now respects the
  generation counter: it keeps current-generation entries and only replaces a
  same-generation entry when the candidate is shallower. Previously a deep current
  entry could be overwritten by a shallow entry from an older search.
- **Root continuation history.** The root move is written to `moveStack[0]` before
  searching its subtree; ply 1 derives its counter-move and continuation-history keys
  from it and was reading a stale slot.

A Stockfish-style capture history was implemented and measured, but a 16-game match
against the same engine without it finished 6-6 with 4 draws, so it was not kept.

## Version 1.3.3 — AVX2 NNUE for x86_64

The Android x86_64 package is compiled against the SSE4.2 baseline, so the
compile-time AVX2 branches in `nnue.cpp` were skipped and its NNUE kernels fell
back to scalar loops. On x86 Android devices (Chromebooks, emulators, WSA) that
made evaluation several times slower than necessary.

`nnue.cpp` now also provides AVX2 clones of the three hot kernels (accumulator
row update, two-perspective row update, output-layer inference) as
`__attribute__((target("avx2")))` functions, selected at runtime with
`__builtin_cpu_supports("avx2")`. The dispatcher is only compiled when the
baseline lacks AVX2, so native AVX2/AVX-512 desktop builds keep their existing
paths. The arm64 NEON path is unchanged.

`bench 12` (722,767 nodes), same sources compiled natively:

| build | time |
|---|---|
| scalar fallback | 4766 ms |
| x86-64 baseline + runtime dispatch | 383 ms |
| `-mavx2` compile-time | 380 ms |

All three produce identical static evaluations on the probe set and an identical
depth-15 search trace and best move, because the AVX2 kernels use the same
saturating int16 arithmetic. The old 1.3.2 x86_64 library contained no AVX2
instructions; the 1.3.3 library contains the kernels and the guarded dispatch.

Two packages are attached to the release (same files under versioned names in `apk/`):

- `wall-e.tiny.apk` — `com.tianl1234.walle`, one network (fingolfin), ~2.2 MB.
- `wall-e.strong.apk` — `com.tianl1234.walle.strong`, three phase networks
  (fingolfin/finarfin/feanor), ~5.9 MB.

Both are versionCode 8 / versionName 1.3.3, signed with the same key as earlier
releases and upgrade in place. The change does not affect ARM phones: they
already ran the NEON kernels.

## Version 1.3.2 — NNUE actually enabled

The startup sanity check in `nnue.cpp` probed the network with bare-kings positions
that are far outside its training distribution: the net genuinely rates a lone bishop
below a pawn there, so the loader rejected a perfectly good network on every startup
and the engine silently played on the much weaker handcrafted evaluation. The check
now probes realistic positions instead (start position near equal, identical eval from
either side to move, and removing a knight from either side swings the eval by roughly
a knight's value), so the embedded network finally loads: the engine reports
`NNUE loaded: <embedded network>`.

Further fixes shipped with this release, all verified against the previous build:

- The transposition-table probe no longer returns a *different position's* stored move
  on a miss (it had been corrupting move ordering and wrongly disabling internal
  iterative reductions at every miss).
- The TT depth field is an `int8_t` and overflowed negative on `go depth > 127`;
  it is now clamped.
- Lazy SMP helpers restarted iterative deepening from depth 1 every iteration; they now
  keep aspiration windows and root ordering across depths.
- SEE rewritten as a threshold swap algorithm — verified equivalent on 19.3M capture
  checks — plus TT prefetching (~34% faster fixed-depth bench) and an ARM NEON
  inference path split across two accumulators to rule out 32-bit lane overflow.

Measured at 200 ms/move, the fixed engine beats the previous (handcrafted-eval) build
39–1 over 40 games. The NNUE evaluation matches Patricia's reference implementation
of the same network bit-exactly on every probed position.

## Version 1.3 — incremental NNUE and packaging

The desktop engine hard-coded BMI2 (`_pext_u64`, `_pdep_u64`) and x86 SIMD headers, so it
could not compile for a phone. These changes fixed that without touching the search:

1. **Portable slider backend.** Magic bitboards with multipliers generated offline
   (`build/gen_magics.cpp`) and embedded as constants. Chosen over a runtime search because
   the hardest squares need hundreds of thousands of attempts, which is far too slow at
   app launch. A collision check at table build time turns a wrong constant into a loud
   failure rather than silent misbehaviour.
   The x86 build still uses PEXT; `-DWALLE_NO_PEXT` forces the portable path for testing.
2. **NEON NNUE.** `nnue.cpp` gained NEON implementations of the three hot kernels
   (saturating accumulator add, batched delta apply, output-layer inference), written to
   match the x86 arithmetic exactly: wrapping int16 products, 32-bit accumulation, same
   rounding. Scalar code remains as the final fallback.
3. **Embedded network.** A GUI executes the engine from the APK's lib directory, where no
   `nnue/` folder exists, so Android builds link the network into the binary
   (`nnue_embedded.cpp`) and load it with `Nnue::load_embedded()`.
4. **Incremental NNUE optimization (v1.3).** Apply each added/removed feature directly to
   each perspective's int16 accumulator with saturating SIMD operations (NEON on ARM64),
   rather than allocating and reducing a two-perspective int32 delta buffer on each move.
5. **PIE.** Android requires position-independent executables; the NDK build produces one.

## Verification done here

- Host build (`-march=znver5`, embedded networks, same sources as the APK): `perft 5` =
  4,865,609, `verify 4` = 197,281 nodes with 0 inconsistent, `bench 12` = 1,153,781 nodes
  in ~339 ms (3.4 Mnps). The node count rose from 556,713 in 1.3.6 because the evaluation
  now reports 56 cp instead of 112 in the start position (see 1.3.7), so eval-driven
  pruning is less optimistic at unchanged speed. Before the qsearch table probe the same
  bench searched 665,358 nodes in ~164 ms, so the reduction costs no strength: a 16-game
  60 ms match between the two builds finished 7-7 with 4 draws.
- Evaluation agrees across the shipped code paths: the AVX-512 build and an SSE4.2 build
  with the runtime AVX2 dispatch used by the x86_64 package return identical scores on
  the same positions, and `nnue.cpp`/`chess.cpp` compile for `arm64-v8a` and `x86_64` in
  both the tiny and the strong variant with NDK 27.3.
- Phase policy (1.3.6): start position and a 16-material-point middlegame auto-select
  fingolfin and give exactly the pinned-fingolfin scores; a 12-point endgame auto-selects
  finarfin and gives exactly the pinned-finarfin scores. The strong build loads all three
  networks; the tiny build loads fingolfin.
- Reference positions: Kiwipete perft(4) = 4,085,603, position 3 perft(5) = 674,624,
  position 4 perft(4) = 422,333, position 5 perft(4) = 2,103,487, position 6 perft(3) =
  89,890, all matching the published counts; `verify 3`/`verify 4` clean on the same
  positions and on replayed self-play games.
- `Threads=2` bench and `verify 3` run clean; the 16-game two-thread match above also
  replayed and verified both engines' final positions.
- Cross-compiled engines: AArch64 and x86-64 ELF `ET_DYN` PIE, entry points set,
  LOAD alignment 16 KiB, no text relocations, only `libc`/`libm`/`libdl` needed;
  the NEON kernels are present (`smull`/`smlal`, `sqadd`/`sqsub`, `addv`) and the
  network's bytes are inside the binary.
- APK (1.3.7): versionCode 12 / versionName 1.3.7, minSdk 26, targetSdk 34,
  `extractNativeLibs=true`, native code `arm64-v8a` + `x86_64`, **no launcher category**
  (so no icon), OEX activity/provider/enginelist present, `zipalign` and `apksigner`
  verify (v2 + v3). The embedded networks were compared byte-for-byte with
  `nnue/*.nnue`: fingolfin in both tiny binaries, all three nets in both strong binaries.
- Windows Subsystem for Android 13 (x86_64) on the 1.3.5 build, carrying the same engine
  and packaging: `perft 5` = 4,865,609, `verify 4` = 197,281 nodes / 0 inconsistent,
  `bench 12` = 556,713 nodes in ~199 ms, a `Threads=4` depth-16 search completes, and the
  strong package loads all three networks and searches across a phase switch. This
  exercises the x86_64 build, not the ARM64 NEON path. The 1.3.6 and 1.3.7 packages were
  not re-run in WSA because its adb endpoint was unavailable at build time; their engine
  changes are the phase policy and the side-to-move correction, both verified on the host
  build above.

Build the current APK with `android/build_apk.sh`; rerun native and package checks after
changes before quoting updated performance or APK checksum results.

## Known limitation

The ARM64 package has still not run on ARM hardware; WSA is x86_64, so the NEON NNUE
kernels remain compile- and structure-verified only. On a phone, the first meaningful
test is `perft 5` from the GUI, which must return 4,865,609.
