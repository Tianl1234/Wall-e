# Wall-E — APK version

This branch is the **Android version** of the Wall-E chess engine: the same C++20 engine,
ported to AArch64 and packaged as an installable APK with the NNUE network embedded.

**Download:** [Wall-E APK version 1.1](https://github.com/Tianl1234/Wall-e/releases/tag/v1.0.0-android) —
`Wall-E-console.apk` (or use the copy in [`apk/`](apk/)).

```
Wall-E-console.apk   1.3 MB, signed, arm64-v8a + x86_64, Android 8.0+
```

Copy it to an arm64 phone (or an x86_64 emulator/Chromebook), tap it, and allow
installing from that source. It needs no permissions, no network and no Google services.

The APK is universal on purpose: an arm64-only build installs on an x86_64 emulator but
never extracts an engine, which shows up as "engine binary missing". The app now prints
the device ABIs and the contents of `nativeLibraryDir` at startup, so a mismatch is
visible instead of silent.

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
apk/Wall-E-console.apk        the built, signed APK (versionCode 2, version 1.1)
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

Output: `../build/Wall-E-console.apk`, signed with a generated development key
(`android/keystore/wall-e.jks`, password `walletee`, alias `walle` — a dev key, not a store
key). Build-tools 34.0.0 cannot dex this app (its D8 crashes on modern javac's anonymous
classes); 36.1.0 is required.

## Verification

| Check | Result |
|---|---|
| Portable slider path (x86, `-DWALLE_NO_PEXT`) | all six standard perft positions pass; `bench 12` = 912,862 nodes, identical to PEXT |
| ARM engine | AArch64, PIE, entry `0x2A7F0`, no TEXTREL, needs only `libc`/`libm`/`libdl` |
| APK manifest | `com.tianl1234.walle`, versionCode 2 / versionName 1.1, minSdk 26, targetSdk 34, `extractNativeLibs=true` |
| APK contents | `assets/nnue/fingolfin.nnue` 1,579,010 B; `lib/arm64-v8a/libengine.so` 559,816 B; `lib/x86_64/libengine.so` 548,600 B |
| APK ABIs | `native-code: 'arm64-v8a' 'x86_64'` |
| Signature | `apksigner verify` exit 0, v2 + v3 true |
| APK size / SHA-256 | 1,340,083 B / `86c235e98c85322e67e05f668b0a9643e1df755835e3673d652ce01edb4623ce` |

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
