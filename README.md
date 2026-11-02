# wall-e android version

This folder is a **separate copy** of the Wall-E chess engine, ported to AArch64 and
packaged as an Android engine APK. The original desktop engine in
`../Chess-engine/Chess-engine` is untouched; nothing here writes back to it.

The APK has **no user interface at all**: no activity, no chessboard, no console, no
launcher icon. It is a headless UCI engine in the format Android chess GUIs discover
(Open Exchange), so it is installed like an engine and appears only in a GUI's engine
list. `android/README.md` documents the protocol and how to import it.

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
VARIANT=tiny   bash build_apk.sh   # -> ../build/wall-e.tiny.apk
VARIANT=strong bash build_apk.sh   # -> ../build/wall-e.strong.apk
```

Both packages contain arm64-v8a and x86_64 code with the network embedded. `tiny`
carries fingolfin only; `strong` adds the finarfin and feanor phase networks. They
use different application ids, so they install side by side.

Requires JDK 17+ and an Android SDK with `platforms;android-34`,
`build-tools;36.1.0` (D8 in 34.0.0 crashes on modern javac output) and
`ndk;27.3.13750724`. No Gradle.

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

Two packages are attached to the release:

- `Wall-E-engine-1.3.3.apk` — `com.tianl1234.walle`, one network (fingolfin), ~2.2 MB.
- `Wall-E-engine-1.3.3-strong.apk` — `com.tianl1234.walle.strong`, three phase
  networks (fingolfin/finarfin/feanor), ~5.9 MB.

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

- Portable slider path on x86 (`-DWALLE_NO_PEXT`): all six standard perft positions pass,
  and  `bench 12` yields exactly the same node count as the PEXT build (912,862 nodes).
- Host build after the load refactor: `NNUE loaded: nnue/fingolfin.nnue`,
  `perft 5` = 4,865,609, `bench 12` = 912,862 nodes.
- Cross-compiled engines: AArch64 and x86-64 ELF `ET_DYN` PIE, entry points set,
  LOAD alignment 16 KiB, no text relocations, only `libc`/`libm`/`libdl` needed;
  the network's bytes are present inside both binaries.
- APKs: `com.tianl1234.walle` and `com.tianl1234.walle.strong`, versionCode 8 /
  versionName 1.3.3, minSdk 26, targetSdk 34, `extractNativeLibs=true`, native code
  `arm64-v8a` + `x86_64`, **no launcher category** (so no icon), OEX
  activity/provider/enginelist present, signature verifies (v2 + v3).
- 1.3.3 x86_64 libraries: `.text` section is identical to a locally rebuilt
  unstripped library, contain the AVX2 kernels and the `have_avx2()` dispatch, and
  embed the networks byte-for-byte (one in `tiny`, three in `strong`).

Build the current APK with `android/build_apk.sh`; rerun native and package checks after
changes before quoting updated performance or APK checksum results.

## Known limitation

There is no ARM device or emulator on this machine, so the ARM package could not be
installed or launched here. The NEON kernels and the extraction/exec path are
compile- and structure-verified only. The x86_64 kernels were additionally executed
natively at the same baseline (see the 1.3.3 table). On a phone, the first meaningful
test is `perft 5` from the GUI, which must return 4,865,609.
