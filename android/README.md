# Wall-E — Android engine package (no UI)

An installable APK that contains the Wall-E chess engine and **nothing else**. There is
no screen, no chessboard, no console, and no launcher icon — it is a headless UCI engine
packaged the same way as every other Android engine (Stockfish, Patricia, Ethereal,
Crafty, …), so a chess GUI can find and use it.

That "invisible after install" behaviour is the point: the package has no
`android.intent.category.LAUNCHER` entry. There is nothing to open; the engine shows up
inside a GUI's engine list instead.

## Why this format

Android 10+ forbids executing binaries from app-writable storage (W^X), so the old
"copy an ARM binary to the SD card and import it" route no longer works. The workaround
every Android engine uses is to ship the engine as a **native library** inside an APK:

```
lib/arm64-v8a/libwalle.so
lib/x86_64/libwalle.so
```

`android:extractNativeLibs="true"` makes Android's package manager extract those files
into the app's `nativeLibraryDir`, which is executable and not app-writable, so GUIs are
allowed to run them. The `.so` name is a packaging requirement; the file itself is a
normal PIE executable with a `main()` that speaks UCI on stdin/stdout.

## Discovery protocol (Open Exchange / OEX)

The package implements the convention that Android chess GUIs use. Everything below was
copied from working engine APKs rather than invented:

| Piece | Value |
|---|---|
| Marker action | `intent.chess.provider.ENGINE` on an exported activity |
| Provider authority | `chess.provider.engine.authority` meta-data on that activity |
| Engine list resource | `res/xml/enginelist.xml`, found via `getIdentifier("enginelist", "xml", pkg)` |
| Executable | `ApplicationInfo.nativeLibraryDir + "/" + filename` |
| Provider MIME type | `application/x-chess-engine` |

`res/xml/enginelist.xml`:

```xml
<enginelist>
    <engine name="Wall-E 1.3.7" filename="libwalle.so" target="arm64-v8a|x86_64" />
</enginelist>
```

A GUI queries activities with the marker action, reads the authority, opens that
package's `enginelist`, picks the `<engine>` whose `target` contains the device ABI and
executes the file in `nativeLibraryDir`. `EngineProvider` additionally serves the binary
over `content://com.tianl1234.walle.EngineProvider/libwalle.so` for GUIs that copy the
engine through a content resolver instead of executing it in place.

## The network is inside the binary

There is no `nnue/fingolfin.nnue` file in the APK. The GUI runs the engine from a
directory it does not control, so a relative path can never be trusted. Instead
`nnue_embedded.cpp` appends the 1,579,010-byte network to the executable with
`.incbin`, and `Nnue::load_embedded()` reads it from there. A GUI log will show:

```
info string NNUE loaded: <embedded network>
```

If instead it says `NNUE unavailable`, the network was not linked — the engine still
plays, but with the weaker handcrafted evaluation.

The strong package embeds three networks (`fingolfin`, `finarfin`, `feanor`) and reports
`+ finarfin + feanor` in the same line. Since 1.3.6 it evaluates with fingolfin and
switches to finarfin only in endgames (half the material or less); feanor is embedded but
no longer selected automatically, because its noise showed up in GUI evaluation bars
during opening analysis. The choice is made once per `go` and the transposition table is
cleared when the active network changes.

Since 1.3.7 the reported score is corrected for the side-to-move bonus the networks carry
(about +1.1 pawns on a symmetric board): `nnue.cpp` measures that bonus on the start
position when the network loads and subtracts half of it, 56 cp for fingolfin, 41 for
finarfin and 60 for feanor. The start position therefore reports `56` instead of `112`,
and consecutive depths no longer disagree by that bonus, which is what stopped GUI
evaluation bars from moving on their own during a review.

## Using it

1. Install `build/wall-e.tiny.apk` (one network) or `build/wall-e.strong.apk` (three phase
   networks, larger). It will not appear in the launcher — by design.
2. In a GUI that supports OEX engines (Chess for Android, Chessis, and other
   kalab-based GUIs), open the engine menu and **import/select an engine**.
3. Pick **Wall-E 1.3.7** from the list.

For Chess for Android specifically: long-press the notation window → *UCI and XBoard* →
*Import engine* → *Wall-E 1.3.7*. No SD-card import step is needed.

The APK upgrades the previous engine package in place (`versionCode 12`, same signing
key), so the old icon disappears when you install it.

Both packages default to `Threads=2` because phone GUIs rarely change engine options;
set `Threads` in the GUI if you want more or fewer. The 256 MB hash default is shared by
both threads.

## Build

Requirements: JDK 17+, Android SDK with `platforms;android-34`,
`build-tools;36.1.0` (D8 in 34.0.0 crashes on modern javac output) and
`ndk;27.3.13750724`. No Gradle, no Android Studio.

```sh
cd android
bash build_apk.sh                 # -> ../build/wall-e.tiny.apk   (one network)
VARIANT=strong bash build_apk.sh  # -> ../build/wall-e.strong.apk (three networks)
```

Toolchain paths can be overridden with `ANDROID_SDK_ROOT`, `ANDROID_NDK_HOME`,
`BUILD_TOOLS_VERSION`, `NDK_VERSION`, `API_LEVEL` and `JAVA_HOME_NIX`. The script fails
the build if a `LAUNCHER` category ever appears in the manifest, so an accidental icon
cannot ship.

### Build gotchas worth knowing

- **build-tools 34.0.0 cannot dex this app.** Its D8 (R8 8.2.2) crashes with an internal
  `NullPointerException` on anonymous inner classes emitted by a modern javac, so the
  script defaults to 36.1.0. Toolchain bug, not a source problem.
- **Assets and native libraries are added with `jar`, not `zip`.** Git Bash has no
  `zip`, and aapt2 on Windows writes asset entries with backslashes, which
  `AssetManager` cannot resolve.
- **The engine is compiled from the project root.** `nnue_embedded.cpp` uses
  `.incbin "nnue/fingolfin.nnue"`, whose path is resolved against the compiler's
  working directory.
- **`-Wl,-z,max-page-size=16384`** keeps the executable loadable on 16 KB-page devices
  (Android 15+ on some ARM64 hardware).

## Verification status

Build with `android/build_apk.sh` and refresh the results below after a successful build.

| Check | Result |
|---|---|
| Launcher entry | none — `aapt2 dump badging` shows no `launchable-activity`, `application: icon=''`; build script enforces it |
| Manifest | `com.tianl1234.walle`, versionCode 12 / versionName 1.3.7, minSdk 26, targetSdk 34, `extractNativeLibs=true` |
| OEX wiring | `intent.chess.provider.ENGINE`, `chess.provider.engine.authority`, exported `EngineProvider`, `res/xml/enginelist.xml` present as `xml/enginelist`, name `Wall-E 1.3.7` |
| Native code (1.3.7) | both ABIs `ET_DYN` PIE, LOAD alignment 16 KiB, no TEXTREL, only `libc`/`libm`/`libdl` needed |
| Embedded network (1.3.7) | byte-for-byte identical to `nnue/*.nnue`: fingolfin in both tiny libraries; fingolfin, finarfin and feanor in both strong libraries |
| Signature (1.3.7) | `apksigner` verifies v2 + v3 with the existing Wall-E key |
| Loader/update checks (1.3.7) | host build from the same sources: `perft 5` = 4,865,609, `verify 4` = 197,281 nodes / 0 inconsistent, `bench 12` = 1,153,781 nodes (the count rose from 556,713 because the start position now evaluates 56 cp instead of 112) |
| APK contents (1.3.7) | manifest, `res/xml/enginelist.xml`, `classes.dex`, `lib/arm64-v8a`, `lib/x86_64`; `zipalign -c -p 4` passes |
| Install (WSA) | 1.3.5 packages installed on Windows Subsystem for Android 13 (x86_64), `primaryCpuAbi=x86_64`, and the OEX query resolved `com.tianl1234.walle/.EngineActivity`; 1.3.6 and 1.3.7 packaging was verified on the host because WSA's adb endpoint was not available at build time |
| Runtime (WSA, 1.3.5) | extracted `libwalle.so` executes from `nativeLibraryDir`-equivalent path: `perft 5` = 4,865,609, `verify 4` = 197,281 nodes / 0 inconsistent, `bench 12` = 556,713 nodes in 197–199 ms (~2.8 Mnps), `Threads=4` depth-16 search completes |
| Phase nets (1.3.7) | host build with the APK's sources: start position and 16-point middlegame auto-select fingolfin and match pinned-fingolfin scores; 12-point endgame auto-selects finarfin and matches pinned-finarfin; strong build embeds all three |

**Not checked:** the ARM64 package has still not run on ARM hardware. WSA is x86_64 and
therefore exercises only the x86_64 build (SSE4.2 baseline with runtime AVX2), not the
NEON NNUE kernels; those remain compile- and structure-verified. The first phone test is
`perft 5` from the GUI — it must return 4,865,609. The provider's `openAssetFile` path is
also unexercised: `adb shell content read` calls `openFile` (which the provider
intentionally does not implement), while OEX GUIs execute the binary from
`nativeLibraryDir` in place.
