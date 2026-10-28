#!/usr/bin/env bash
# Builds build/Wall-E-console.apk without Gradle.
#
# Pipeline: NDK clang++ (native engine, one build per ABI) -> javac -> d8 ->
# aapt2 -> jar -> zipalign -> apksigner. Every tool is from the Android SDK /
# JDK, so the build needs no Maven downloads and no Android Studio.
#
# The APK is universal: arm64 phones use arm64-v8a, x86_64 emulators and
# Chromebooks use x86_64. A missing ABI is the failure mode where the app
# installs but then reports that the engine binary is missing.
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
ABIS=("arm64-v8a" "x86_64")

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
APK="$PROJECT/build/Wall-E-console.apk"

die() { echo "error: $*" >&2; exit 1; }

[[ -x "$BT/aapt2.exe" || -f "$BT/aapt2.exe" ]] || die "aapt2 not found in $BT"
[[ -f "$PLATFORM" ]] || die "android.jar not found at $PLATFORM"
[[ -d "$NDK" ]] || die "NDK not found at $NDK"
[[ -x "$JAVAC" || -f "$JAVAC" ]] || die "javac not found at $JAVAC"
[[ -f "$PROJECT/chess.cpp" && -f "$PROJECT/nnue.cpp" ]] || die "engine sources missing"

rm -rf "$OUT"
mkdir -p "$OUT/classes" "$OUT/dex" "$OUT/stage" "$PROJECT/build"

echo "== 1/7 native engine (API $API: ${ABIS[*]})"
for ABI in "${ABIS[@]}"; do
    case "$ABI" in
        arm64-v8a) TARGET="aarch64-linux-android${API}" ;;
        x86_64)    TARGET="x86_64-linux-android${API}" ;;
        *) die "unsupported ABI $ABI" ;;
    esac
    CXX="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/${TARGET}-clang++.cmd"
    [[ -f "$CXX" ]] || CXX="${CXX%.cmd}"
    [[ -f "$CXX" ]] || die "NDK clang++ for $TARGET not found"
    echo "   $ABI"
    mkdir -p "$OUT/lib/$ABI"
    "$CXX" -O3 -std=c++20 -DNDEBUG -fPIE -pie -static-libstdc++ \
           -Wl,--gc-sections -s \
           -o "$OUT/lib/$ABI/libengine.so" \
           "$PROJECT/chess.cpp" "$PROJECT/nnue.cpp"
done

echo "== 2/7 java sources"
# Quoted globs (not $(find ...)) so the space in this folder's name survives.
"$JAVAC" -source 8 -target 8 -Xlint:-options -classpath "$PLATFORM" \
         -d "$OUT/classes" "$HERE/java/com/tianl1234/walle"/*.java

echo "== 3/7 dex"
"$BT/d8.bat" --min-api "$API" --lib "$PLATFORM" \
    --output "$OUT/dex" "$OUT/classes/com/tianl1234/walle"/*.class

echo "== 4/7 resources"
"$BT/aapt2.exe" compile --dir "$HERE/res" -o "$OUT/res.zip"

echo "== 5/7 stage assets (NNUE network)"
# Assets and native libs are added with jar rather than aapt2 -A/zip: this shell
# has no zip, and aapt2 on Windows writes asset entries with a backslash
# separator, which AssetManager cannot open as "nnue/fingolfin.nnue".
mkdir -p "$OUT/stage/assets/nnue" "$OUT/stage/lib"
cp "$PROJECT/nnue/fingolfin.nnue" "$OUT/stage/assets/nnue/fingolfin.nnue"
cp "$OUT/dex/classes.dex" "$OUT/stage/"
for ABI in "${ABIS[@]}"; do
    mkdir -p "$OUT/stage/lib/$ABI"
    cp "$OUT/lib/$ABI/libengine.so" "$OUT/stage/lib/$ABI/"
done

echo "== 6/7 package + link"
"$BT/aapt2.exe" link -o "$OUT/base.apk" -I "$PLATFORM" \
    --manifest "$HERE/AndroidManifest.xml" \
    --min-sdk-version "$API" --target-sdk-version "$TARGET_API" \
    "$OUT/res.zip"

(cd "$OUT/stage" && "$JAR" uf "$OUT/base.apk" classes.dex assets lib)

"$BT/zipalign.exe" -f -p 4 "$OUT/base.apk" "$OUT/aligned.apk"

echo "== 7/7 sign"
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
"$BT/apksigner.bat" verify --print-certs "$APK" | grep -E "^(Verifies|Signer #1 certificate DN)" || true

echo
echo "built: $APK ($(du -h "$APK" | cut -f1))"
