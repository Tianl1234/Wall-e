# Wall-E — APK version

This branch is the **Android version** of the Wall-E chess engine: the same C++20 engine,
ported to AArch64 and packaged as an installable APK with the NNUE network embedded.

**Download:** [Wall-E APK version 1](https://github.com/Tianl1234/Wall-e/releases/tag/v1.0.0-android) —
`Wall-E-console-arm64.apk` (or use the copy in [`apk/`](apk/)).

```
Wall-E-console-arm64.apk   1.1 MB, signed, arm64-v8a, Android 8.0+
```

Copy it to an arm64 phone, tap it, and allow installing from that source. It needs no
permissions, no network and no Google services.

## What it is

A **UCI console**, not a board GUI: it starts the engine, streams every line the engine
prints, and offers presets for the engine's own commands.

- **Bench** — `bench 12`, six standard positions, prints nodes/time/NPS
- **Perft 5** — move-generation self-test from the FEN field (empty = startpos)
- **Analyze** — `go depth 16` from the FEN field, streaming the PV
- **Stop / Restart / Clear**, a FEN field and a free-text UCI command box
  (`setoption name Threads value 4`, `go movetime 5000`, `eval`, `verify 3`, …)

On launch it extracts the network to app storage and prints the engine's own
`info string NNUE loaded: …` line. If that line says `NNUE unavailable`, the engine is
running on the weaker handcrafted evaluation.

## Contents

```
chess.cpp, nnue.cpp, nnue.h   engine sources, ported to AArch64
build.sh                      desktop (x86-64) build, unchanged behaviour
nnue/fingolfin.nnue           evaluation network (MIT, Adam Kulju)
android/                      app sources, manifest, Gradle-free build script
apk/Wall-E-console-arm64.apk  the built, signed APK
```

## Android port — what changed

The desktop engine was x86-only; three things had to change, none of them in the search:

1. **Portable slider backend.** `_pext_u64`/`_pdep_u64` are BMI2 instructions. On AArch64 the
   engine uses magic bitboards with multipliers generated offline
   (`android/…` notes; generator is in the local Android project) and embedded as constants,
   because a runtime search needs hundreds of thousands of attempts on the hard squares and
   would stall app startup. A collision check turns a bad constant into a loud failure.
   x86 keeps PEXT; `-DWALLE_NO_PEXT` forces the portable path for testing.
2. **NEON NNUE.** Guarded the x86 intrinsics and added NEON kernels for the three hot
   operations: saturating accumulator add, batched delta apply, and output-layer inference.
   The arithmetic mirrors the x86 path exactly (wrapping int16 products, 32-bit accumulation,
   same rounding).
3. **Packaging.** Android requires PIE executables and blocks executing files in app storage
   on Android 10+, so the engine ships as `lib/arm64-v8a/libengine.so` with
   `extractNativeLibs="true"`; `PackageManager` extracts it executable.

## Build the APK

Requires JDK 17+, Android SDK `platforms;android-34`, `build-tools;36.1.0` and
`ndk;27.3.13750724`. No Gradle.

```sh
cd android
bash build_apk.sh
```

Output: `../build/Wall-E-console-arm64.apk`, signed with a generated development key
(`android/keystore/wall-e.jks`, password `walletee`, alias `walle` — a dev key, not a store
key). Build-tools 34.0.0 cannot dex this app (its D8 crashes on modern javac's anonymous
classes); 36.1.0 is required.

## Verification

| Check | Result |
|---|---|
| Portable slider path (x86, `-DWALLE_NO_PEXT`) | all six standard perft positions pass; `bench 12` = 912,862 nodes, identical to PEXT |
| ARM engine | AArch64, PIE, entry `0x2A7F0`, no TEXTREL, needs only `libc`/`libm`/`libdl` |
| APK manifest | `com.tianl1234.walle`, minSdk 26, targetSdk 34, `extractNativeLibs=true` |
| APK contents | `assets/nnue/fingolfin.nnue` 1,579,010 B; `lib/arm64-v8a/libengine.so` 559,816 B |
| Signature | `apksigner verify` exit 0, v2 + v3 true |
| APK SHA-256 | `8ab99b6d080fed48870335b350f30bf0c8f4d9488a9734d629e1097a0107f18f` |

Not verified: the app has never been launched. There is no ARM device, emulator or `adb` in
the build environment, so the NEON kernels are compile-verified only. On the phone, tap
**Perft 5** with an empty FEN field first — startpos perft(5) must be **4,865,609**.

## Strength

Unrated, same as the desktop engine: measured 1.20 Mnps on the six-position bench and depth 14
from startpos in 143 ms on a Ryzen 9 9900X (single thread). Estimated ~2400 Elo class at
blitz; no match against rated opposition has been run.

## License

MIT for the engine (see [LICENSE](LICENSE)). The network `nnue/fingolfin.nnue` is Patricia's
by Adam Kulju, used under MIT — see [nnue/PATRICIA-LICENSE.txt](nnue/PATRICIA-LICENSE.txt).
