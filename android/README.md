# Wall-E — Android console

An installable APK (version 1.1) that runs the Wall-E chess engine on Android, with the
NNUE network embedded inside the APK. It is a **console**, not a board GUI: it runs the
engine, streams every UCI line it prints, and offers presets for the engine's own test
commands (bench, perft, analysis).

The APK is **universal**: `arm64-v8a` for phones and `x86_64` for emulators and
Chromebooks. An APK that ships only one ABI installs on other devices but never extracts
an engine, which is exactly the "engine binary missing" failure this build fixes; the
app now also prints the device ABIs and the contents of `nativeLibraryDir` at startup.

## What is in the APK

| Part | Contents |
|---|---|
| Engine | `lib/arm64-v8a/libengine.so` and `lib/x86_64/libengine.so` — the same C++ engine per ABI |
| Network | `assets/nnue/fingolfin.nnue` (extracted to app storage on first launch) |
| UI | One activity: transcript, FEN field, command field, presets |

The engine is shipped as `libengine.so` on purpose. Android extracts native libraries into
an executable directory, while files written to the app's private data directory cannot be
executed at all on Android 10+ (W^X). Copying the binary from assets and `chmod +x`-ing it
would therefore fail with `EACCES` on any modern phone. `android:extractNativeLibs="true"`
in the manifest is what guarantees the extraction happens.

## Porting notes

The engine was x86-only. Three things had to change for AArch64:

1. **Slider attacks.** `_pext_u64` / `_pdep_u64` are BMI2 instructions. The portable backend
   uses magic bitboards with multipliers generated offline (see `build/gen_magics.cpp`) and
   embedded as constants — searching for them at runtime needs hundreds of thousands of
   attempts on the hardest squares, which is far too slow at app start. A collision check
   turns a bad constant into a loud startup failure instead of silently wrong moves.
   On x86 the PEXT backend is still used, and `-DWALLE_NO_PEXT` forces the portable path so
   it can be tested on a desktop.
2. **NNUE.** The x86 intrinsics are guarded, and `nnue.cpp` gained NEON paths for the three
   hot operations: saturating accumulator adds (`vqaddq_s16`), batched delta application
   (widening `vmull`/`vaddq_s32` + saturating `vqmovn_s32`) and output-layer inference
   (`vminq`/`vmaxq` clamp, wrapping `vmulq_s16`, `vmlal_s16` accumulation). The arithmetic
   mirrors the x86 path exactly — same wrapping int16 products, same 32-bit accumulation,
   same rounding.
3. **PIE.** Android requires position-independent executables; the NDK build uses `-fPIE -pie`.

Everything else (search, TT, Lazy SMP, UCI loop, `perft`/`verify`/`bench`) is untouched.

## Build

Requirements:

- JDK 17+ (the script uses `JAVA_HOME_NIX`, defaulting to the local JDK 21)
- Android SDK with `platforms;android-34`, `build-tools;36.1.0`, `ndk;27.3.13750724`
- No Gradle, no Android Studio, no Maven downloads

```sh
cd "android"
bash build_apk.sh
```

Output: `build/Wall-E-console.apk`, signed with a generated development key
(`android/keystore/wall-e.jks`, password `walletee`, alias `walle`). That key is fine for
sideloading; a Play Store upload would need your own keystore.

Toolchain paths can be overridden with `ANDROID_SDK_ROOT`, `ANDROID_NDK_HOME`,
`BUILD_TOOLS_VERSION`, `NDK_VERSION`, `API_LEVEL` and `JAVA_HOME_NIX`.

### Two build gotchas worth knowing

- **build-tools 34.0.0 cannot dex this app.** Its D8 (R8 8.2.2) crashes with an internal
  `NullPointerException` on every anonymous inner class that a modern javac emits, so the
  script defaults to 36.1.0. This is a toolchain bug, not a source problem.
- **Assets and the native library are added with `jar`, not `zip`/`aapt2 -A`.** Git Bash has
  no `zip`, and aapt2 on Windows stores asset entries with a backslash
  (`assets/nnue\fingolfin.nnue`), which `AssetManager.open("nnue/fingolfin.nnue")` cannot
  resolve. Staging them and updating the APK with the JDK's `jar` keeps forward slashes.

## Install

Copy the APK to the phone (USB, Drive, anything), tap it, and allow installing from that
source when Android asks. It needs no permissions, no network and no Google services.
If the engine is reported missing, the transcript now shows the device ABIs and the
`nativeLibraryDir` contents, which identifies an ABI mismatch immediately.

## Using it

- On launch the network is extracted, the engine starts, and the transcript shows
  `info string NNUE loaded: /data/user/0/com.tianl1234.walle/files/nnue/fingolfin.nnue`.
  That line is the proof the network is active; if it says `NNUE unavailable`, the app is
  running on the handcrafted fallback evaluation.
- **Bench** runs `bench 12` (six standard positions) and prints nodes/time/NPS.
- **Perft 5** sets the position from the FEN field (empty = startpos) and runs perft.
- **Analyze** sets the position and runs `go depth 16`, streaming the PV.
- **Stop**, **Restart** and **Clear** do the obvious things.
- The command field accepts any UCI command, e.g. `setoption name Threads value 4`,
  `go movetime 5000`, `eval`, `verify 3`.

## Verification status

Checked on this machine:

| Check | Result |
|---|---|
| Portable slider backend (x86, `-DWALLE_NO_PEXT`) | six standard perft positions pass; `bench 12` = 912,862 nodes, identical to the PEXT build |
| NNUE | `NNUE loaded` on both builds; `eval` returns a value |
| Native engine ELF | arm64: AArch64 PIE, entry `0x2A7F0`; x86_64: PIE, entry `0x29730`; no TEXTREL, only `libc`/`libm`/`libdl` needed |
| APK manifest | `com.tianl1234.walle`, versionCode 2 / versionName 1.1, `minSdkVersion 26`, `targetSdkVersion 34`, `extractNativeLibs=true` |
| APK ABIs | `native-code: 'arm64-v8a' 'x86_64'` |
| APK contents | `classes.dex` 17,840 B; `assets/nnue/fingolfin.nnue` 1,579,010 B; `lib/arm64-v8a/libengine.so` 559,816 B; `lib/x86_64/libengine.so` 548,600 B |
| Signature | `apksigner verify`: exit 0, v2 and v3 schemes true |
| APK size / SHA-256 | 1,340,083 B / `86c235e98c85322e67e05f668b0a9643e1df755835e3673d652ce01edb4623ce` |

Not checked: the app was never launched. This machine has no ARM CPU, no emulator and no
`adb`, so the NEON kernels are compile-verified only. If they were wrong the engine would
misplay rather than crash, which is why one **Perft 5** tap on the phone is the first thing
to do — the transcript prints the node count, so a wrong slider backend or accumulator shows
up immediately.
