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
bash build_apk.sh
# -> ../build/Wall-E-engine.apk  (arm64-v8a + x86_64, network embedded)
```

Requires JDK 17+ and an Android SDK with `platforms;android-34`,
`build-tools;36.1.0` (D8 in 34.0.0 crashes on modern javac output) and
`ndk;27.3.13750724`. No Gradle.

## Engine changes for ARM

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
4. **PIE.** Android requires position-independent executables; the NDK build produces one.

## Verification done here

- Portable slider path on x86 (`-DWALLE_NO_PEXT`): all six standard perft positions pass,
  and `bench 12` yields exactly the same node count as the PEXT build (912,862 nodes).
- Host build after the load refactor: `NNUE loaded: nnue/fingolfin.nnue`,
  `perft 5` = 4,865,609, `bench 12` = 912,862 nodes.
- Cross-compiled engines: AArch64 and x86-64 ELF `ET_DYN` PIE, entry points set,
  LOAD alignment 16 KiB, no text relocations, only `libc`/`libm`/`libdl` needed;
  the network's bytes are present inside both binaries.
- APK: `com.tianl1234.walle`, versionCode 3 / versionName 1.2, minSdk 26, targetSdk 34,
  `extractNativeLibs=true`, native code `arm64-v8a` + `x86_64`, **no launcher category**
  (so no icon), OEX activity/provider/enginelist present, signature verifies (v2 + v3).

Built APK: `build/Wall-E-engine.apk` — 2,253,409 bytes, signed,
SHA-256 `ed280b7649fbf344b805280ecbaa0477a04f09aeb302d628d1ab95264fdeb098`.

## Known limitation

There is no ARM device, emulator or `adb` on this machine, so the package could not be
installed or launched here. The NEON kernels and the extraction/exec path are
compile- and structure-verified only. On the phone, the first meaningful test is
`perft 5` from the GUI, which must return 4,865,609.
