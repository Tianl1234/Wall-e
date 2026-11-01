// Embeds the NNUE network(s) in the engine binary itself.
//
// Android executes the engine straight out of the APK's nativeLibraryDir, so there is
// no reliable working directory and no way for the engine to read assets. Baking the
// network in removes the problem: the networks become part of libwalle.so and
// `Nnue::load_embedded()` (nnue.cpp) reads them from these symbols.
//
// .incbin is used instead of a generated C array on purpose: a 1.5 million number
// initializer makes the compiler allocate hundreds of MB and adds a minute to every
// build, while .incbin copies the bytes verbatim into .rodata.
//
// The paths are relative to the compiler's working directory, so build_apk.sh compiles
// this file from the project root, where the nnue/ folder lives.
//
// Two packages share this file. Both embed fingolfin (the middlegame net); the
// "strong" package additionally embeds finarfin and feanor and picks between the
// three per search (see WALLE_PHASE_NETS in chess.cpp).

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
#if defined(WALLE_PHASE_NETS)
    ".balign 64\n"
    ".global walle_nnue_endgame_start\n"
    ".hidden walle_nnue_endgame_start\n"
    "walle_nnue_endgame_start:\n"
    ".incbin \"nnue/finarfin.nnue\"\n"
    ".global walle_nnue_endgame_end\n"
    ".hidden walle_nnue_endgame_end\n"
    "walle_nnue_endgame_end:\n"
    ".balign 64\n"
    ".global walle_nnue_aggressive_start\n"
    ".hidden walle_nnue_aggressive_start\n"
    "walle_nnue_aggressive_start:\n"
    ".incbin \"nnue/feanor.nnue\"\n"
    ".global walle_nnue_aggressive_end\n"
    ".hidden walle_nnue_aggressive_end\n"
    "walle_nnue_aggressive_end:\n"
#endif
    ".previous\n");
