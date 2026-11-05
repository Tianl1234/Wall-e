#!/usr/bin/env bash
# Builds build/Wall-E-engine.apk: a chess ENGINE package with no user interface.
#
# It follows the Open Exchange (OEX) convention used by Android engine packages
# (Stockfish, Patricia, Ethereal, ...): no launcher activity, an activity advertising
# intent.chess.provider.ENGINE, res/xml/enginelist.xml, and the engine itself as
# lib/<abi>/libwalle.so so the package manager extracts it into nativeLibraryDir.
#
# Pipeline: NDK clang++ (engine, NNUE embedded) -> javac -> d8 -> aapt2 -> jar ->
# zipalign -> apksigner. No Gradle, no Android Studio, no Maven downloads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

SDK="${ANDROID_SDK_ROOT:-/c/Users/Administrator/AppData/Local/Android/Sdk}"
# 34.0.0 ships R8 8.2.2, whose D8 crashes on anonymous inner classes produced by
# a modern javac; 36.1.0 has a working D8.
BUILD_TOOLS_VERSION="${BUILD_TOOLS_VERSION:-36.1.0}"
NDK_VERSION="${NDK_VERSION:-27.3.13750724}"
API="${API_LEVEL:-26}"
TARGET_API=34
ABIS=(arm64-v8a x86_64)

BT="$SDK/build-tools/$BUILD_TOOLS_VERSION"
PLATFORM="$SDK/platforms/android-$TARGET_API/android.jar"
NDK="${ANDROID_NDK_HOME:-$SDK/ndk/$NDK_VERSION}"

JAVA_HOME_NIX="${JAVA_HOME_NIX:-/c/Program Files/Microsoft/jdk-21.0.11.10-hotspot}"
export JAVA_HOME="$(cygpath -w "$JAVA_HOME_NIX" 2>/dev/null || echo "$JAVA_HOME_NIX")"
export PATH="$JAVA_HOME_NIX/bin:$PATH"
JAVAC="$JAVA_HOME_NIX/bin/javac.exe"
KEYTOOL="$JAVA_HOME_NIX/bin/keytool.exe"
JAR="$JAVA_HOME_NIX/bin/jar.exe"

STORE_PASS="walletee"
KEY_PASS="walletee"
KEY_ALIAS="walle"

OUT="$PROJECT/build/apk"

die() { echo "error: $*" >&2; exit 1; }

# Both packages are built from the same sources; the variant decides how many
# networks are embedded and under which application id it is published, so the two
# can be installed side by side.
#
#   tiny   - one network (fingolfin), the package published so far
#   strong - three networks, one per game phase (see WALLE_PHASE_NETS in chess.cpp)
VARIANT="${VARIANT:-tiny}"
case "$VARIANT" in
    tiny)
        APP_ID="com.tianl1234.walle"
        APP_LABEL="Wall-E Chess Engine"
        ENGINE_NAME="Wall-E 1.3.7"
        EXTRA_CXXFLAGS=()
        APK_NAME="wall-e.tiny.apk"
        ;;
    strong)
        APP_ID="com.tianl1234.walle.strong"
        APP_LABEL="Wall-E Chess Engine (Strong)"
        ENGINE_NAME="Wall-E 1.3.7 Strong"
        EXTRA_CXXFLAGS=(-DWALLE_PHASE_NETS)
        APK_NAME="wall-e.strong.apk"
        ;;
    *) die "unknown VARIANT '$VARIANT' (expected tiny or strong)" ;;
esac
APK="$PROJECT/build/$APK_NAME"

[[ -f "$BT/aapt2.exe" ]] || die "aapt2 not found in $BT"
[[ -f "$PLATFORM" ]] || die "android.jar not found at $PLATFORM"
[[ -d "$NDK" ]] || die "NDK not found at $NDK"
[[ -f "$JAVAC" ]] || die "javac not found at $JAVAC"
[[ -f "$PROJECT/chess.cpp" && -f "$PROJECT/nnue.cpp" && -f "$PROJECT/nnue_embedded.cpp" ]] \
    || die "engine sources missing"
[[ -f "$PROJECT/nnue/fingolfin.nnue" ]] || die "network nnue/fingolfin.nnue missing"
if [[ "$VARIANT" == "strong" ]]; then
    [[ -f "$PROJECT/nnue/finarfin.nnue" && -f "$PROJECT/nnue/feanor.nnue" ]] \
        || die "the strong variant also needs nnue/finarfin.nnue and nnue/feanor.nnue"
fi

clang_for() {
    local triple="$1"
    local cxx="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/${triple}${API}-clang++.cmd"
    [[ -f "$cxx" ]] || cxx="${cxx%.cmd}"
    [[ -f "$cxx" ]] || die "NDK clang++ for $triple (API $API) not found"
    printf '%s' "$cxx"
}

rm -rf "$OUT"
mkdir -p "$OUT/classes" "$OUT/dex" "$OUT/stage" "$PROJECT/build"

# The manifest and the engine list differ per variant (application id, label, engine
# name), so they are generated here instead of being duplicated as source files.
mkdir -p "$OUT/res/xml"
sed -e "s|package=\"com.tianl1234.walle\"|package=\"$APP_ID\"|" \
    -e "s|android:authorities=\"com.tianl1234.walle.EngineProvider\"|android:authorities=\"$APP_ID.EngineProvider\"|" \
    -e "s|android:value=\"com.tianl1234.walle.EngineProvider\"|android:value=\"$APP_ID.EngineProvider\"|" \
    -e "s|android:label=\"Wall-E Chess Engine\"|android:label=\"$APP_LABEL\"|" \
    "$HERE/AndroidManifest.xml" > "$OUT/AndroidManifest.xml"
sed -e "s|name=\"Wall-E[^\"]*\"|name=\"$ENGINE_NAME\"|" \
    "$HERE/res/xml/enginelist.xml" > "$OUT/res/xml/enginelist.xml"

echo "== 1/6 native engine (embedded NNUE, PIE, max-page-size 16 KiB)"
# Compiled from the project root: nnue_embedded.cpp uses .incbin "nnue/fingolfin.nnue",
# which the assembler resolves against the compiler's working directory.
for abi in "${ABIS[@]}"; do
    case "$abi" in
        arm64-v8a) triple="aarch64-linux-android" ;;
        x86_64)    triple="x86_64-linux-android" ;;
        *) die "unsupported ABI $abi" ;;
    esac
    CXX="$(clang_for "$triple")"
    echo "   $abi"
    mkdir -p "$OUT/stage/lib/$abi"
    # -flto lets the search inline the evaluation across translation units;
    # -ffunction-sections/-fdata-sections is what makes --gc-sections effective;
    # -fno-exceptions/-fno-rtti drop code the engine never uses.
    # -fomit-frame-pointer: the NDK keeps the frame pointer by default on arm64.
    # The baseline ISA is deliberately left at the default (armv8-a): a -mcpu such
    # as cortex-a76 would emit LSE/dotprod instructions and crash on older
    # API 26 devices. Tune per device only if the minimum device is known.
    ( cd "$PROJECT" && "$CXX" -O3 -std=c++20 -DNDEBUG -DWALLE_EMBEDDED_NET \
            ${EXTRA_CXXFLAGS[@]+"${EXTRA_CXXFLAGS[@]}"} \
            -flto -fno-exceptions -fno-rtti -fomit-frame-pointer \
            -ffunction-sections -fdata-sections \
            -fPIE -pie -static-libstdc++ \
            -Wl,--gc-sections -Wl,-z,max-page-size=16384 -s \
            -o "$OUT/stage/lib/$abi/libwalle.so" \
            chess.cpp nnue.cpp nnue_embedded.cpp )
done

echo "== 2/6 java sources (provider only, no UI)"
"$JAVAC" -source 8 -target 8 -Xlint:-options -classpath "$PLATFORM" \
         -d "$OUT/classes" "$HERE/java/com/tianl1234/walle"/*.java

echo "== 3/6 dex"
"$BT/d8.bat" --min-api "$API" --lib "$PLATFORM" \
    --output "$OUT/dex" "$OUT/classes/com/tianl1234/walle"/*.class
cp "$OUT/dex/classes.dex" "$OUT/stage/"

echo "== 4/6 resources (res/xml/enginelist.xml)"
"$BT/aapt2.exe" compile --dir "$OUT/res" -o "$OUT/res.zip"

echo "== 5/6 package + link"
"$BT/aapt2.exe" link -o "$OUT/base.apk" -I "$PLATFORM" \
    --manifest "$OUT/AndroidManifest.xml" \
    --min-sdk-version "$API" --target-sdk-version "$TARGET_API" \
    "$OUT/res.zip"

# jar (not zip) because this shell has none; it also keeps forward slashes in the
# lib/ entries, which native library extraction requires.
( cd "$OUT/stage" && "$JAR" uf "$OUT/base.apk" classes.dex lib )

"$BT/zipalign.exe" -f -p 4 "$OUT/base.apk" "$OUT/aligned.apk"

echo "== 6/6 sign"
KS="$HERE/keystore/wall-e.jks"
if [[ ! -f "$KS" ]]; then
    mkdir -p "$HERE/keystore"
    "$KEYTOOL" -genkeypair -keystore "$KS" \
        -storepass "$STORE_PASS" -keypass "$KEY_PASS" -alias "$KEY_ALIAS" \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Wall-E, OU=Android, O=Tianl1234, C=DE" >/dev/null 2>&1
    echo "   generated development keystore: $KS"
fi

"$BT/apksigner.bat" sign \
    --ks "$KS" --ks-pass "pass:$STORE_PASS" --key-pass "pass:$KEY_PASS" \
    --ks-key-alias "$KEY_ALIAS" \
    --out "$APK" "$OUT/aligned.apk"

echo
echo "== verify"
"$BT/aapt2.exe" dump badging "$APK" | grep -E "^(package|sdkVersion|targetSdkVersion|application-label|native-code)" || true

# Enforce the whole point of this package: no launcher entry, so no icon and no UI.
if "$BT/aapt2.exe" dump xmltree --file AndroidManifest.xml "$APK" | grep -q 'android.intent.category.LAUNCHER'; then
    die "APK declares a LAUNCHER category - it would show an icon"
fi
echo "   no launcher category (no icon, no UI): ok"

"$BT/apksigner.bat" verify --print-certs "$APK" | grep -E "^(Verifies|Signer #1 certificate DN)" || true

echo
echo "built: $APK ($(du -h "$APK" | cut -f1))"
