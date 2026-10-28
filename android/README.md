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
    <engine name="Wall-E 1.2" filename="libwalle.so" target="arm64-v8a|x86_64" />
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

## Using it

1. Install `Wall-E-engine.apk` (it will not appear in the launcher — by design).
2. In a GUI that supports OEX engines (Chess for Android, Chessis, and other
   kalab-based GUIs), open the engine menu and **import/select an engine**.
3. Pick **Wall-E 1.2** from the list.

For Chess for Android specifically: long-press the notation window → *UCI and XBoard* →
*Import engine* → *Wall-E 1.2*. No SD-card import step is needed.

The APK upgrades the previous console build in place (`versionCode 3`, same signing
key), so the old icon disappears when you install it.

## Build

Requirements: JDK 17+, Android SDK with `platforms;android-34`,
`build-tools;36.1.0` (D8 in 34.0.0 crashes on modern javac output) and
`ndk;27.3.13750724`. No Gradle, no Android Studio.

```sh
cd android
bash build_apk.sh
# -> ../build/Wall-E-engine.apk  (arm64-v8a + x86_64)
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

Checked on this machine (`build/Wall-E-engine.apk`, 2,253,409 bytes,
SHA-256 `ed280b7649fbf344b805280ecbaa0477a04f09aeb302d628d1ab95264fdeb098`):

| Check | Result |
|---|---|
| Launcher entry | none — `aapt2 dump badging` shows no `launchable-activity`, `application: icon=''`; build script enforces it |
| Manifest | `com.tianl1234.walle`, versionCode 3 / versionName 1.2, minSdk 26, targetSdk 34, `extractNativeLibs=true` |
| OEX wiring | `intent.chess.provider.ENGINE`, `chess.provider.engine.authority`, exported `EngineProvider`, `res/xml/enginelist.xml` present as `xml/enginelist` |
| Native code | `arm64-v8a`, `x86_64`; both ELF64 `ET_DYN` PIE with entry points, LOAD `Align 0x4000`, only `libc`/`libm`/`libdl` needed, no TEXTREL |
| Embedded network | network's first 64 bytes found inside both binaries (offsets 73,664 / 71,488); each is 1.58 MB larger than the pre-embedding engine |
| Signature | `apksigner verify`: v2 and v3 true, `CN=Wall-E` |
| Loader refactor (host x86-64) | `NNUE loaded: nnue/fingolfin.nnue`, `perft 5` = **4,865,609**, `bench 12` = 912,862 nodes |
| APK contents | `classes.dex` 3,612 B; `res/xml/enginelist.xml` 352 B; two `libwalle.so`; nothing else except signatures |

**Not checked:** the package was never installed or run on an Android device (no ARM
hardware, no emulator, no adb here). The NEON NNUE kernels and the extraction/exec path
are therefore compile- and structure-verified only. The first real test is `perft 5`
from the GUI — it must return 4,865,609.
