// Embeds the NNUE network in the engine binary itself.
//
// Android executes the engine straight out of the APK's nativeLibraryDir, so there is
// no reliable working directory and no way for the engine to read assets. Baking the
// network in removes the problem: the 1.5 MB network becomes part of libwalle.so and
// `Nnue::load_embedded()` (nnue.cpp) reads it from these symbols.
//
// .incbin is used instead of a generated C array on purpose: a 1.5 million number
// initializer makes the compiler allocate hundreds of MB and adds a minute to every
// build, while .incbin copies the bytes verbatim into .rodata.
//
// The path is relative to the compiler's working directory, so build_apk.sh compiles
// this file from the project root, where nnue/fingolfin.nnue lives.

asm(
    ".section .rodata\n"
    ".balign 64\n"
    ".global walle_nnue_start\n"
    ".hidden walle_nnue_start\n"
    "walle_nnue_start:\n"
    ".incbin \"nnue/fingolfin.nnue\"\n"
    ".global walle_nnue_end\n"
    ".hidden walle_nnue_end\n"
    "walle_nnue_end:\n"
    ".previous\n");
