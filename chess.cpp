// ============================================================================
//  Wall-E 2 — compact UCI chess engine
//
//  Build:  g++ -O3 -march=znver5 -std=c++20 -o chess.exe chess.cpp
//
//  Highlights
//   * bitboards, PEXT sliders (Zen 3+ has fast PEXT/PDEP)
//   * pseudo-legal generation + make/unmake + king-safety test
//   * 16-bit moves, zero allocation in the search hot path
//   * incremental material / PSQT / phase / Zobrist
//   * PVS + aspiration, clustered & aged TT, null move, reverse futility,
//     razoring, ProbCut, IIR, LMR, LMP, futility, SEE pruning, killers,
//     butterfly + continuation history, counter moves, check extension
//   * incremental Patricia NNUE evaluation with a handcrafted fallback
//   * Lazy SMP
// ============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <filesystem>
#include "nnue.h"

// PEXT/PDEP exist only on x86 with BMI2. Every other target (notably AArch64)
// uses the portable magic-bitboard backend below. -DWALLE_NO_PEXT forces the
// portable path on x86 as well, which is how it gets tested on the desktop.
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__)) && !defined(WALLE_NO_PEXT)
#define WALLE_PEXT 1
#else
#define WALLE_PEXT 0
#endif

using U64 = uint64_t;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
//  Types
// ---------------------------------------------------------------------------
enum : int { WHITE = 0, BLACK = 1 };
enum : int { PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
enum : int { EMPTY = 0 };
enum : int { W_P = 1, W_N, W_B, W_R, W_Q, W_K, B_P = 9, B_N, B_B, B_R, B_Q, B_K };

inline int make_pc(int c, int t) { return (c << 3) | t; }
inline int pc_type(int p)        { return p & 7; }
inline int pc_color(int p)       { return p >> 3; }

enum : int {
    MT_QUIET = 0, MT_DOUBLE = 1, MT_CAPTURE = 2, MT_EP = 3, MT_CASTLE = 4,
    MT_PROMO_N = 5, MT_PROMO_B = 6, MT_PROMO_R = 7, MT_PROMO_Q = 8,
    MT_PROMO_CAP_N = 9, MT_PROMO_CAP_B = 10, MT_PROMO_CAP_R = 11, MT_PROMO_CAP_Q = 12
};

inline int mv_from(uint16_t m) { return m & 63; }
inline int mv_to(uint16_t m)   { return (m >> 6) & 63; }
inline int mv_type(uint16_t m) { return m >> 12; }
inline bool is_capture(int mt) { return mt == MT_CAPTURE || mt == MT_EP || mt >= MT_PROMO_CAP_N; }
inline bool is_promo(int mt)   { return mt >= MT_PROMO_N; }
inline int  promo_type(int mt) { return ((mt - MT_PROMO_N) & 3) + KNIGHT; }
inline uint16_t make_move(int from, int to, int mt) {
    return (uint16_t)(from | (to << 6) | (mt << 12));
}

constexpr int MAX_PLY     = 128;
constexpr int MAX_MOVES   = 256;
constexpr int INF         = 32000;
constexpr int MATE        = 31000;
constexpr int MATE_IN_MAX = MATE - MAX_PLY;
constexpr int VALUE_NONE  = 32500;
constexpr int MAX_HISTORY = 2048;

inline int popcount(U64 b) { return __builtin_popcountll(b); }
inline int lsb(U64 b)      { return __builtin_ctzll(b); }
inline U64 bit(int s)      { return 1ULL << s; }
inline int file_of(int s)  { return s & 7; }
inline int rank_of(int s)  { return s >> 3; }

// ---------------------------------------------------------------------------
//  Tables
// ---------------------------------------------------------------------------
U64 PawnAtt[2][64];
U64 KnightAtt[64], KingAtt[64];
U64 RookMask[64], BishopMask[64];
U64 RookTable[64][4096];
U64 BishopTable[64][512];
#if !WALLE_PEXT
int RookShift[64], BishopShift[64];
#endif
U64 CastleMask[64];

constexpr U64 FILE_A_BB = 0x0101010101010101ULL;
constexpr U64 FILE_H_BB = 0x8080808080808080ULL;
constexpr U64 RANK_1_BB = 0x00000000000000FFULL;
constexpr U64 RANK_2_BB = 0x000000000000FF00ULL;
constexpr U64 RANK_7_BB = 0x00FF000000000000ULL;
constexpr U64 RANK_8_BB = 0xFF00000000000000ULL;

U64 FileBB[8], AdjFilesBB[8];
U64 RankAbove[8], RankBelow[8];
U64 PassedMask[2][64];
U64 CenterBB, SpaceBB;

U64 Zpiece[16][64], Zside, Zcastle[16], Zep[8];

int PSQ_MG[16][64], PSQ_EG[16][64];
int PhaseInc[16];
int LMRTable[64][64];

const int MG_VAL[7] = {0, 82, 337, 365, 477, 1025, 0};
const int EG_VAL[7] = {0, 94, 281, 297, 512,  936, 0};
const int PH_INC[7] = {0, 0, 1, 1, 2, 4, 0};

// PeSTO, rank 8 first (index 0 == a8)
static const int MG_PAWN_T[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int EG_PAWN_T[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int MG_KNIGHT_T[64] = {
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
};
static const int EG_KNIGHT_T[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};
static const int MG_BISHOP_T[64] = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};
static const int EG_BISHOP_T[64] = {
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
static const int MG_ROOK_T[64] = {
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
};
static const int EG_ROOK_T[64] = {
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
};
static const int MG_QUEEN_T[64] = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};
static const int EG_QUEEN_T[64] = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};
static const int MG_KING_T[64] = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};
static const int EG_KING_T[64] = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

const int* MG_TBL[7] = {nullptr, MG_PAWN_T, MG_KNIGHT_T, MG_BISHOP_T, MG_ROOK_T, MG_QUEEN_T, MG_KING_T};
const int* EG_TBL[7] = {nullptr, EG_PAWN_T, EG_KNIGHT_T, EG_BISHOP_T, EG_ROOK_T, EG_QUEEN_T, EG_KING_T};

// ---------------------------------------------------------------------------
//  Init
// ---------------------------------------------------------------------------
static U64 slide_attacks(int sq, U64 occ, const int dirs[4][2]) {
    U64 att = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int d = 0; d < 4; ++d) {
        int nr = r + dirs[d][0], nf = f + dirs[d][1];
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
            int s = nr * 8 + nf;
            att |= bit(s);
            if (occ & bit(s)) break;
            nr += dirs[d][0]; nf += dirs[d][1];
        }
    }
    return att;
}

static U64 pext_mask(int sq, const int dirs[4][2]) {
    U64 m = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int d = 0; d < 4; ++d) {
        int nr = r + dirs[d][0], nf = f + dirs[d][1];
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
            int nnr = nr + dirs[d][0], nnf = nf + dirs[d][1];
            if (nnr < 0 || nnr > 7 || nnf < 0 || nnf > 7) break;   // edge square not included
            m |= bit(nr * 8 + nf);
            nr = nnr; nf = nnf;
        }
    }
    return m;
}

static U64 next_rand() {
    static U64 s = 0x97643210FEDCBA98ULL;
    U64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

#if !WALLE_PEXT
// Scatter the low bits of idx into the set bits of mask: the inverse of PEXT.
static U64 scatter_bits(int idx, U64 mask) {
    U64 occ = 0;
    while (mask) {
        const int b = lsb(mask);
        mask &= mask - 1;
        if (idx & 1) occ |= bit(b);
        idx >>= 1;
    }
    return occ;
}

// Magic multipliers generated offline with this exact mask convention
// (shift = 64 - popcount(mask)) by build/gen_magics.cpp. Embedding them keeps
// startup instant: searching for them at runtime needs hundreds of thousands
// of attempts on the hardest squares, which is far too slow on a phone.
static const U64 ROOK_MAGICS[64] = {
    0x948001A084400011ULL, 0x0040100040002002ULL, 0x0080200080100008ULL, 0x1100042100081001ULL,
    0x0200100200080420ULL, 0x0B00040032281100ULL, 0x8480090000801600ULL, 0x0600008044110022ULL,
    0xC803800040026380ULL, 0x50C0402000401002ULL, 0x8080801000802000ULL, 0x8402001042000820ULL,
    0x5400800800800400ULL, 0x5022000200100804ULL, 0x000C000190420804ULL, 0x8830800100004080ULL,
    0x4000908000400020ULL, 0x0810004040002009ULL, 0x8830002004080021ULL, 0x4048090021001000ULL,
    0x0898008008800400ULL, 0x0200808002000400ULL, 0x0010040002010810ULL, 0x00000200011040A4ULL,
    0x0380004040002000ULL, 0x4000200080804000ULL, 0x0900200080100080ULL, 0x0000100080080080ULL,
    0x8100040080800800ULL, 0x0100040080020080ULL, 0x8000020400011088ULL, 0x8042488600050044ULL,
    0x2081C00284800020ULL, 0x0030102000404000ULL, 0x0120010041002018ULL, 0x0500800800801004ULL,
    0x5020040080800800ULL, 0x0000800200800400ULL, 0x8411100884000241ULL, 0x0008004102000084ULL,
    0x2000204000908000ULL, 0x0020200050004000ULL, 0x0100200010008080ULL, 0x0094100009010020ULL,
    0x4808000402004040ULL, 0x2881002400090022ULL, 0x4200020004010100ULL, 0x0000408044020001ULL,
    0x0B18402900800100ULL, 0x1040830240002300ULL, 0x0020200010008080ULL, 0x0002100021000900ULL,
    0x0300800400080080ULL, 0x0006000410080200ULL, 0x0401000200040100ULL, 0x0A04010040840200ULL,
    0x2008902140800301ULL, 0x0044810018244001ULL, 0x2001002000084011ULL, 0x2042208910005501ULL,
    0x00020020481004AAULL, 0x0201000816040047ULL, 0x001000C102100804ULL, 0x0582105100208402ULL,
};

static const U64 BISHOP_MAGICS[64] = {
    0x4090300090841040ULL, 0x0060240400982100ULL, 0x0804214202040800ULL, 0x1804410020150C01ULL,
    0x0804504109822042ULL, 0x00008211408500A0ULL, 0x8402022202420082ULL, 0x0204240202108208ULL,
    0x00080909100C0840ULL, 0x000012101E022040ULL, 0x080604880A004000ULL, 0x3600844100200200ULL,
    0x0024020210883840ULL, 0x0200020184200000ULL, 0x00001C22080404C0ULL, 0x8080060209110811ULL,
    0x0420000461020210ULL, 0x0420900401120220ULL, 0x0004000808202202ULL, 0x0010410808102022ULL,
    0x8604010210220200ULL, 0x0004089210040402ULL, 0x2840800404444204ULL, 0x0400422084240100ULL,
    0x0820902005042804ULL, 0x0008840648010820ULL, 0x0880481010002040ULL, 0x0311080004040410ULL,
    0x1501010118104000ULL, 0x43100410A0820080ULL, 0x0000820901211024ULL, 0x0404010334210148ULL,
    0x01010410C3421000ULL, 0xA302701000040107ULL, 0x0382010113100050ULL, 0x8100080800060A00ULL,
    0x0080420020020080ULL, 0xA082080108621000ULL, 0x2110810100005400ULL, 0x2000810208004203ULL,
    0x01084222A0001001ULL, 0x8000840420048260ULL, 0x00008A0802001401ULL, 0x8100810280802802ULL,
    0x006002020C000200ULL, 0x100401040101480AULL, 0xA084040800404231ULL, 0x0010211048920100ULL,
    0x0041041044841000ULL, 0x4302005242300000ULL, 0x0504002884103000ULL, 0x4040521042020808ULL,
    0x02082072A0220805ULL, 0x1090083010023000ULL, 0x11101012B8104000ULL, 0x02A0260202670802ULL,
    0xA010404800882000ULL, 0x0200009414010402ULL, 0x0004004202010448ULL, 0x0A00005000420210ULL,
    0x4000002C48830400ULL, 0x000284882088108AULL, 0x0802400902208200ULL, 0x0068022088021084ULL,
};

// Fill one square's attack table. The collision check turns a wrong constant
// into an immediate, loud failure instead of silently wrong moves.
static void fill_magic_table(int sq, U64 mask, U64 magic, int shift,
                             const int dirs[4][2], U64* table) {
    const int size = 1 << popcount(mask);
    for (int i = 0; i < size; ++i) {
        const U64 occ = scatter_bits(i, mask);
        const U64 att = slide_attacks(sq, occ, dirs);
        U64& slot = table[(occ * magic) >> shift];
        if (slot && slot != att) {
            std::cerr << "fatal: magic collision for square " << sq << '\n';
            std::exit(1);
        }
        slot = att;
    }
}
#endif

void init_all() {
    for (int s = 0; s < 64; ++s) {
        int r = rank_of(s), f = file_of(s);
        U64 bb = 0;
        static const int ndr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
        static const int ndf[] = {-1, 1, -2, 2, -2, 2, -1, 1};
        for (int i = 0; i < 8; ++i) {
            int nr = r + ndr[i], nf = f + ndf[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) bb |= bit(nr * 8 + nf);
        }
        KnightAtt[s] = bb;

        bb = 0;
        static const int kdr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        static const int kdf[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        for (int i = 0; i < 8; ++i) {
            int nr = r + kdr[i], nf = f + kdf[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) bb |= bit(nr * 8 + nf);
        }
        KingAtt[s] = bb;

        PawnAtt[WHITE][s] = 0;
        PawnAtt[BLACK][s] = 0;
        if (r < 7) {
            if (f > 0) PawnAtt[WHITE][s] |= bit(s + 7);
            if (f < 7) PawnAtt[WHITE][s] |= bit(s + 9);
        }
        if (r > 0) {
            if (f > 0) PawnAtt[BLACK][s] |= bit(s - 9);
            if (f < 7) PawnAtt[BLACK][s] |= bit(s - 7);
        }
    }

    static const int rook_dirs[4][2]   = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static const int bishop_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int s = 0; s < 64; ++s) {
        RookMask[s]   = pext_mask(s, rook_dirs);
        BishopMask[s] = pext_mask(s, bishop_dirs);
#if WALLE_PEXT
        int rn = 1 << popcount(RookMask[s]);
        for (int i = 0; i < rn; ++i)
            RookTable[s][i] = slide_attacks(s, _pdep_u64((U64)i, RookMask[s]), rook_dirs);
        int bn = 1 << popcount(BishopMask[s]);
        for (int i = 0; i < bn; ++i)
            BishopTable[s][i] = slide_attacks(s, _pdep_u64((U64)i, BishopMask[s]), bishop_dirs);
#else
        RookShift[s]   = 64 - popcount(RookMask[s]);
        BishopShift[s] = 64 - popcount(BishopMask[s]);
        fill_magic_table(s, RookMask[s], ROOK_MAGICS[s], RookShift[s], rook_dirs, RookTable[s]);
        fill_magic_table(s, BishopMask[s], BISHOP_MAGICS[s], BishopShift[s], bishop_dirs, BishopTable[s]);
#endif
    }

    for (int c = 0; c < 2; ++c)
        for (int p = 1; p <= 6; ++p)
            for (int s = 0; s < 64; ++s) {
                int idx = (c == WHITE) ? (s ^ 56) : s;
                PSQ_MG[make_pc(c, p)][s] = MG_VAL[p] + MG_TBL[p][idx];
                PSQ_EG[make_pc(c, p)][s] = EG_VAL[p] + EG_TBL[p][idx];
            }

    for (int p = 0; p < 16; ++p) PhaseInc[p] = 0;
    for (int p = 1; p <= 6; ++p) PhaseInc[p] = PH_INC[p];
    for (int p = 9; p <= 14; ++p) PhaseInc[p] = PH_INC[p & 7];

    Zpiece[0][0] = 0;
    for (int s = 0; s < 64; ++s) {
        Zpiece[0][s] = 0;
        for (int p = 1; p <= 14; ++p) Zpiece[p][s] = next_rand();
    }
    Zside = next_rand();
    for (int i = 0; i < 16; ++i) Zcastle[i] = next_rand();
    for (int i = 0; i < 8; ++i)  Zep[i] = next_rand();

    for (int f = 0; f < 8; ++f) {
        FileBB[f] = FILE_A_BB << f;
        U64 adj = 0;
        if (f > 0) adj |= FileBB[f - 1];
        if (f < 7) adj |= FileBB[f + 1];
        AdjFilesBB[f] = adj;
    }
    for (int r = 0; r < 8; ++r) {
        RankAbove[r] = 0;
        RankBelow[r] = 0;
        for (int rr = r + 1; rr < 8; ++rr) RankAbove[r] |= 0xFFULL << (8 * rr);
        for (int rr = 0; rr < r; ++rr)     RankBelow[r] |= 0xFFULL << (8 * rr);
    }
    for (int s = 0; s < 64; ++s) {
        U64 files = FileBB[file_of(s)] | AdjFilesBB[file_of(s)];
        PassedMask[WHITE][s] = files & RankAbove[rank_of(s)];
        PassedMask[BLACK][s] = files & RankBelow[rank_of(s)];
    }
    CenterBB = (FileBB[3] | FileBB[4]) & ((0xFFULL << 24) | (0xFFULL << 32));
    SpaceBB  = (FileBB[2] | FileBB[3] | FileBB[4] | FileBB[5]) &
               (0xFFULL << 8 | 0xFFULL << 16 | 0xFFULL << 24 | 0xFFULL << 32);

    for (int s = 0; s < 64; ++s) CastleMask[s] = ~0ULL;
    CastleMask[4]  = ~(1ULL | 2ULL);
    CastleMask[0]  = ~2ULL;
    CastleMask[7]  = ~1ULL;
    CastleMask[60] = ~(4ULL | 8ULL);
    CastleMask[56] = ~8ULL;
    CastleMask[63] = ~4ULL;
    CastleMask[4] &= 0xF;
    // values are used as bit-masks on the 4-bit rights field
    for (int s = 0; s < 64; ++s) CastleMask[s] &= 0xF;

    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            LMRTable[d][m] = (int)(0.77 + std::log((double)d) * std::log((double)m) / 2.36);
}

#if WALLE_PEXT
inline U64 rook_att(int s, U64 occ)   { return RookTable[s][_pext_u64(occ, RookMask[s])]; }
inline U64 bishop_att(int s, U64 occ) { return BishopTable[s][_pext_u64(occ, BishopMask[s])]; }
#else
inline U64 rook_att(int s, U64 occ)   { return RookTable[s][((occ & RookMask[s]) * ROOK_MAGICS[s]) >> RookShift[s]]; }
inline U64 bishop_att(int s, U64 occ) { return BishopTable[s][((occ & BishopMask[s]) * BISHOP_MAGICS[s]) >> BishopShift[s]]; }
#endif

// ---------------------------------------------------------------------------
//  Position
// ---------------------------------------------------------------------------
struct Undo {
    U64 key;
    int mg, eg, phaseInc;
    uint8_t castling, ep, fifty, captured;
    uint16_t move;
};

struct Position {
    U64 bb[16];
    U64 byType[7];
    U64 occ[2];
    U64 all;
    uint8_t sq[64];
    int stm;
    int castling;
    int ep;
    int fifty;
    int mg, eg, phaseInc;
    U64 key;
    U64 keyHist[MAX_HISTORY];
    int histSize;
    Nnue::Accumulator nnue;
};

inline void add_piece(Position& p, int pc, int s) {
    U64 b = bit(s);
    p.bb[pc] |= b;
    p.byType[pc_type(pc)] |= b;
    p.occ[pc >> 3] |= b;
    p.all |= b;
    p.sq[s] = (uint8_t)pc;
    const int sgn = (pc_color(pc) == WHITE) ? 1 : -1;
    p.mg += sgn * PSQ_MG[pc][s];
    p.eg += sgn * PSQ_EG[pc][s];
    p.phaseInc += PhaseInc[pc];
    p.key ^= Zpiece[pc][s];
}

inline void remove_piece(Position& p, int pc, int s) {
    U64 b = bit(s);
    p.bb[pc] ^= b;
    p.byType[pc_type(pc)] ^= b;
    p.occ[pc >> 3] ^= b;
    p.all ^= b;
    p.sq[s] = EMPTY;
    const int sgn = (pc_color(pc) == WHITE) ? 1 : -1;
    p.mg -= sgn * PSQ_MG[pc][s];
    p.eg -= sgn * PSQ_EG[pc][s];
    p.phaseInc -= PhaseInc[pc];
    p.key ^= Zpiece[pc][s];
}

inline U64 ep_key(const Position& p, int stm, int ep) {
    if (ep < 0) return 0;
    if (PawnAtt[stm ^ 1][ep] & p.bb[make_pc(stm, PAWN)]) return Zep[file_of(ep)];
    return 0;
}

inline void make_move(Position& p, uint16_t m, Undo& u, bool updateNnue = true) {
    const int from = mv_from(m), to = mv_to(m), mt = mv_type(m);
    const int us = p.stm, them = us ^ 1;
    const int pc = p.sq[from];
    Nnue::PieceChange nnueChanges[4];
    int nnueChangeCount = 0;
    auto recordNnue = [&](int piece, int square, bool add) {
        if (updateNnue) nnueChanges[nnueChangeCount++] = {pc_color(piece), pc_type(piece), square, add};
    };

    u.key = p.key;
    u.castling = (uint8_t)p.castling;
    u.ep = (uint8_t)(p.ep < 0 ? 64 : p.ep);
    u.fifty = (uint8_t)p.fifty;
    u.mg = p.mg; u.eg = p.eg; u.phaseInc = p.phaseInc;
    u.move = m;
    u.captured = EMPTY;

    p.key ^= ep_key(p, us, p.ep);
    p.key ^= Zcastle[p.castling];

    int captured = EMPTY, capsq = to;
    if (mt == MT_EP) { capsq = (us == WHITE) ? to - 8 : to + 8; captured = p.sq[capsq]; }
    else if (is_capture(mt)) captured = p.sq[to];
    u.captured = (uint8_t)captured;

    if (captured != EMPTY) { recordNnue(captured, capsq, false); remove_piece(p, captured, capsq); }

    recordNnue(pc, from, false);
    remove_piece(p, pc, from);
    const int placedPc = is_promo(mt) ? make_pc(us, promo_type(mt)) : pc;
    recordNnue(placedPc, to, true);
    if (is_promo(mt)) add_piece(p, placedPc, to);
    else              add_piece(p, pc, to);

    if (mt == MT_CASTLE) {
        int rf, rt;
        if (to == 6)       { rf = 7;  rt = 5;  }
        else if (to == 2)  { rf = 0;  rt = 3;  }
        else if (to == 62) { rf = 63; rt = 61; }
        else               { rf = 56; rt = 59; }
        int rook = make_pc(us, ROOK);
        recordNnue(rook, rf, false);
        recordNnue(rook, rt, true);
        remove_piece(p, rook, rf);
        add_piece(p, rook, rt);
    }

    if (updateNnue) Nnue::apply_changes(p.nnue, nnueChanges, nnueChangeCount);
    p.fifty = (captured != EMPTY || pc_type(pc) == PAWN) ? 0 : p.fifty + 1;
    p.ep = (mt == MT_DOUBLE) ? ((from + to) >> 1) : -1;
    p.castling &= (int)(CastleMask[from] & CastleMask[to]);
    p.stm = them;
    p.key ^= Zside;
    p.key ^= ep_key(p, them, p.ep);
    p.key ^= Zcastle[p.castling];

    if (p.histSize < MAX_HISTORY) p.keyHist[p.histSize] = p.key;
    p.histSize++;
}

inline void unmake(Position& p, const Undo& u, int movedPc, bool updateNnue = true) {
    const uint16_t m = u.move;
    const int from = mv_from(m), to = mv_to(m), mt = mv_type(m);
    const int us = p.stm ^ 1;

    p.histSize--;

    Nnue::PieceChange nnueChanges[4];
    int nnueChangeCount = 0;
    auto recordNnue = [&](int piece, int square, bool add) {
        if (updateNnue) nnueChanges[nnueChangeCount++] = {pc_color(piece), pc_type(piece), square, add};
    };
    if (is_promo(mt)) {
        recordNnue(make_pc(us, promo_type(mt)), to, false);
        remove_piece(p, make_pc(us, promo_type(mt)), to);
    } else {
        recordNnue(movedPc, to, false);
        remove_piece(p, movedPc, to);
    }

    if (mt == MT_CASTLE) {
        int rf, rt;
        if (to == 6)       { rf = 7;  rt = 5;  }
        else if (to == 2)  { rf = 0;  rt = 3;  }
        else if (to == 62) { rf = 63; rt = 61; }
        else               { rf = 56; rt = 59; }
        int rook = make_pc(us, ROOK);
        recordNnue(rook, rt, false);
        recordNnue(rook, rf, true);
        remove_piece(p, rook, rt);
        add_piece(p, rook, rf);
    }

    recordNnue(movedPc, from, true);
    add_piece(p, movedPc, from);

    if (u.captured != EMPTY) {
        int capsq = to;
        if (mt == MT_EP) capsq = (us == WHITE) ? to - 8 : to + 8;

        recordNnue(u.captured, capsq, true);
        add_piece(p, u.captured, capsq);
    }

    if (updateNnue) Nnue::apply_changes(p.nnue, nnueChanges, nnueChangeCount);
    // scalars last: the add/remove calls above touched key/mg/eg/phaseInc
    p.stm = us;
    p.key = u.key;
    p.mg = u.mg;
    p.eg = u.eg;
    p.phaseInc = u.phaseInc;
    p.castling = u.castling;
    p.ep = (u.ep == 64) ? -1 : (int)u.ep;
    p.fifty = u.fifty;
}

// ---------------------------------------------------------------------------
//  Attacks / legality
// ---------------------------------------------------------------------------
inline bool attacked_by(const Position& p, int s, int c) {
    if (PawnAtt[c ^ 1][s] & p.bb[make_pc(c, PAWN)]) return true;
    if (KnightAtt[s] & p.bb[make_pc(c, KNIGHT)]) return true;
    if (KingAtt[s] & p.bb[make_pc(c, KING)]) return true;
    U64 occ = p.all;
    if (bishop_att(s, occ) & (p.bb[make_pc(c, BISHOP)] | p.bb[make_pc(c, QUEEN)])) return true;
    if (rook_att(s, occ) & (p.bb[make_pc(c, ROOK)] | p.bb[make_pc(c, QUEEN)])) return true;
    return false;
}

inline int king_sq(const Position& p, int c) { return lsb(p.bb[make_pc(c, KING)]); }
inline bool in_check(const Position& p) { return attacked_by(p, king_sq(p, p.stm), p.stm ^ 1); }
inline bool illegal(const Position& p)  { return attacked_by(p, king_sq(p, p.stm ^ 1), p.stm); }

inline U64 attackers_to(const Position& p, int s, U64 occ) {
    return (PawnAtt[BLACK][s] & p.bb[W_P]) | (PawnAtt[WHITE][s] & p.bb[B_P]) |
           (KnightAtt[s] & (p.bb[W_N] | p.bb[B_N])) |
           (KingAtt[s] & (p.bb[W_K] | p.bb[B_K])) |
           (rook_att(s, occ) & (p.bb[W_R] | p.bb[B_R] | p.bb[W_Q] | p.bb[B_Q])) |
           (bishop_att(s, occ) & (p.bb[W_B] | p.bb[B_B] | p.bb[W_Q] | p.bb[B_Q]));
}

// ---------------------------------------------------------------------------
//  Move generation
// ---------------------------------------------------------------------------
struct MoveList {
    uint16_t m[MAX_MOVES];
    int n;
    MoveList() : n(0) {}
    inline void add(uint16_t x) { if (n < MAX_MOVES) m[n++] = x; }
};

template <bool OnlyCaps>
static inline void gen_pawns(const Position& p, MoveList& ml) {
    const int us = p.stm;
    const U64 pawns = p.bb[make_pc(us, PAWN)];
    if (!pawns) return;
    const U64 empty = ~p.all;
    const U64 enemies = p.occ[us ^ 1];

    if (us == WHITE) {
        U64 push = (pawns << 8) & empty;
        U64 promoPush = push & RANK_8_BB;
        while (promoPush) {
            int to = lsb(promoPush); promoPush &= promoPush - 1;
            for (int t = MT_PROMO_N; t <= MT_PROMO_Q; ++t) ml.add(make_move(to - 8, to, t));
        }
        if (!OnlyCaps) {
            U64 q = push & ~RANK_8_BB;
            while (q) { int to = lsb(q); q &= q - 1; ml.add(make_move(to - 8, to, MT_QUIET)); }
            U64 dbl = ((pawns & RANK_2_BB) << 16) & (empty << 8) & empty;
            while (dbl) { int to = lsb(dbl); dbl &= dbl - 1; ml.add(make_move(to - 16, to, MT_DOUBLE)); }
        }
        U64 capL = ((pawns & ~FILE_A_BB) << 7) & enemies;
        U64 capR = ((pawns & ~FILE_H_BB) << 9) & enemies;
        U64 pl = capL & RANK_8_BB;
        while (pl) {
            int to = lsb(pl); pl &= pl - 1;
            for (int t = MT_PROMO_CAP_N; t <= MT_PROMO_CAP_Q; ++t) ml.add(make_move(to - 7, to, t));
        }
        U64 pr = capR & RANK_8_BB;
        while (pr) {
            int to = lsb(pr); pr &= pr - 1;
            for (int t = MT_PROMO_CAP_N; t <= MT_PROMO_CAP_Q; ++t) ml.add(make_move(to - 9, to, t));
        }
        capL &= ~RANK_8_BB;
        while (capL) { int to = lsb(capL); capL &= capL - 1; ml.add(make_move(to - 7, to, MT_CAPTURE)); }
        capR &= ~RANK_8_BB;
        while (capR) { int to = lsb(capR); capR &= capR - 1; ml.add(make_move(to - 9, to, MT_CAPTURE)); }
    } else {
        U64 push = (pawns >> 8) & empty;
        U64 promoPush = push & RANK_1_BB;
        while (promoPush) {
            int to = lsb(promoPush); promoPush &= promoPush - 1;
            for (int t = MT_PROMO_N; t <= MT_PROMO_Q; ++t) ml.add(make_move(to + 8, to, t));
        }
        if (!OnlyCaps) {
            U64 q = push & ~RANK_1_BB;
            while (q) { int to = lsb(q); q &= q - 1; ml.add(make_move(to + 8, to, MT_QUIET)); }
            U64 dbl = ((pawns & RANK_7_BB) >> 16) & (empty >> 8) & empty;
            while (dbl) { int to = lsb(dbl); dbl &= dbl - 1; ml.add(make_move(to + 16, to, MT_DOUBLE)); }
        }
        U64 capL = ((pawns & ~FILE_A_BB) >> 9) & enemies;
        U64 capR = ((pawns & ~FILE_H_BB) >> 7) & enemies;
        U64 pl = capL & RANK_1_BB;
        while (pl) {
            int to = lsb(pl); pl &= pl - 1;
            for (int t = MT_PROMO_CAP_N; t <= MT_PROMO_CAP_Q; ++t) ml.add(make_move(to + 9, to, t));
        }
        U64 pr = capR & RANK_1_BB;
        while (pr) {
            int to = lsb(pr); pr &= pr - 1;
            for (int t = MT_PROMO_CAP_N; t <= MT_PROMO_CAP_Q; ++t) ml.add(make_move(to + 7, to, t));
        }
        capL &= ~RANK_1_BB;
        while (capL) { int to = lsb(capL); capL &= capL - 1; ml.add(make_move(to + 9, to, MT_CAPTURE)); }
        capR &= ~RANK_1_BB;
        while (capR) { int to = lsb(capR); capR &= capR - 1; ml.add(make_move(to + 7, to, MT_CAPTURE)); }
    }

    if (p.ep >= 0) {
        U64 epc = PawnAtt[us ^ 1][p.ep] & pawns;
        while (epc) { int from = lsb(epc); epc &= epc - 1; ml.add(make_move(from, p.ep, MT_EP)); }
    }
}

template <bool OnlyCaps>
static inline void gen_all(const Position& p, MoveList& ml) {
    const int us = p.stm;
    const U64 own = p.occ[us];
    const U64 enemies = p.occ[us ^ 1];
    const U64 target = OnlyCaps ? enemies : ~own;
    const U64 occ = p.all;

    gen_pawns<OnlyCaps>(p, ml);

    auto add_moves = [&](int from, U64 att) {
        while (att) {
            int to = lsb(att); att &= att - 1;
            ml.add(make_move(from, to, (enemies & bit(to)) ? MT_CAPTURE : MT_QUIET));
        }
    };

    U64 kn = p.bb[make_pc(us, KNIGHT)];
    while (kn) { int s = lsb(kn); kn &= kn - 1; add_moves(s, KnightAtt[s] & target); }

    U64 bi = p.bb[make_pc(us, BISHOP)];
    while (bi) { int s = lsb(bi); bi &= bi - 1; add_moves(s, bishop_att(s, occ) & target); }

    U64 rk = p.bb[make_pc(us, ROOK)];
    while (rk) { int s = lsb(rk); rk &= rk - 1; add_moves(s, rook_att(s, occ) & target); }

    U64 qu = p.bb[make_pc(us, QUEEN)];
    while (qu) { int s = lsb(qu); qu &= qu - 1; add_moves(s, (bishop_att(s, occ) | rook_att(s, occ)) & target); }

    if (p.bb[make_pc(us, KING)]) {
        int s = king_sq(p, us);
        add_moves(s, KingAtt[s] & target);

        if (!OnlyCaps) {
            if (us == WHITE) {
                if ((p.castling & 1) && !(p.all & 0x60ULL) &&
                    !attacked_by(p, 4, BLACK) && !attacked_by(p, 5, BLACK))
                    ml.add(make_move(4, 6, MT_CASTLE));
                if ((p.castling & 2) && !(p.all & 0x0EULL) &&
                    !attacked_by(p, 4, BLACK) && !attacked_by(p, 3, BLACK))
                    ml.add(make_move(4, 2, MT_CASTLE));
            } else {
                if ((p.castling & 4) && !(p.all & 0x6000000000000000ULL) &&
                    !attacked_by(p, 60, WHITE) && !attacked_by(p, 61, WHITE))
                    ml.add(make_move(60, 62, MT_CASTLE));
                if ((p.castling & 8) && !(p.all & 0x0E00000000000000ULL) &&
                    !attacked_by(p, 60, WHITE) && !attacked_by(p, 59, WHITE))
                    ml.add(make_move(60, 58, MT_CASTLE));
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  SEE
// ---------------------------------------------------------------------------
constexpr int SEE_VAL[7] = {0, 100, 320, 330, 500, 950, 20000};

static int see_rec(const Position& p, int to, int side, int target, U64 occ) {
    U64 att = attackers_to(p, to, occ) & p.occ[side] & occ;
    int pt;
    U64 bb = 0;
    for (pt = PAWN; pt <= KING; ++pt) {
        bb = att & p.bb[make_pc(side, pt)];
        if (bb) break;
    }
    if (!bb) return 0;
    U64 nb = bb & (~bb + 1);
    U64 occ2 = occ ^ nb;
    if (pt == KING) {
        if (attackers_to(p, to, occ2) & p.occ[side ^ 1] & occ2) return 0;
    }
    int reply = see_rec(p, to, side ^ 1, SEE_VAL[pt], occ2);
    return std::max(0, target - reply);
}

int see_value(const Position& p, uint16_t m) {
    const int mt = mv_type(m);
    if (!is_capture(mt)) {
        if (is_promo(mt)) return SEE_VAL[promo_type(mt)] - SEE_VAL[PAWN];
        return 0;
    }
    const int from = mv_from(m), to = mv_to(m);
    const int us = p.stm, them = us ^ 1;
    int gain = SEE_VAL[pc_type(mt == MT_EP ? make_pc(them, PAWN) : p.sq[to])];
    int ourVal = SEE_VAL[pc_type(p.sq[from])];
    if (is_promo(mt)) {
        gain += SEE_VAL[promo_type(mt)] - SEE_VAL[PAWN];
        ourVal = SEE_VAL[promo_type(mt)];
    }
    U64 occ = p.all ^ bit(from);
    if (mt == MT_EP) occ ^= bit(us == WHITE ? to - 8 : to + 8);
    else occ ^= bit(to);
    return gain - see_rec(p, to, them, ourVal, occ);
}

inline bool see_ge(const Position& p, uint16_t m, int threshold) {
    return see_value(p, m) >= threshold;
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------
static const int MobMg[7] = {0, 0, 4, 4, 3, 1, 0};
static const int MobEg[7] = {0, 0, 4, 5, 4, 2, 0};

static const int PassedMg[8] = {0, 5, 10, 20, 40, 70, 110, 0};
static const int PassedEg[8] = {0, 10, 18, 30, 60, 105, 160, 0};

static const int IsolatedMg = -10, IsolatedEg = -13;
static const int DoubledMg  = -9,  DoubledEg  = -22;
static const int BackwardMg = -8,  BackwardEg = -9;
static const int SupportedMg = 5,  SupportedEg = 3;

static const int RookOpenMg = 26, RookOpenEg = 12;
static const int RookSemiMg = 12, RookSemiEg = 8;
static const int Rook7thMg  = 18, Rook7thEg  = 32;
static const int BishopPairMg = 26, BishopPairEg = 42;

static const int ThreatPawnMinorMg = 45, ThreatPawnMinorEg = 25;
static const int ThreatPawnMajorMg = 30, ThreatPawnMajorEg = 18;
static const int ThreatMinorMajorMg = 22, ThreatMinorMajorEg = 14;
static const int ThreatLooseMg = 12, ThreatLooseEg = 8;

inline void hist_bonus(int16_t& h, int bonus) {
    int v = (int)h + bonus - (int)h * std::abs(bonus) / 16384;
    h = (int16_t)std::clamp(v, -16384, 16384);
}
inline void hist_malus(int16_t& h, int bonus) {
    int v = (int)h - bonus + (int)h * std::abs(bonus) / 16384;
    h = (int16_t)std::clamp(v, -16384, 16384);
}

// per attacking piece (pawn attacks are counted per attacked zone square)
static const int KingAttWeight[7] = {0, 0, 4, 3, 4, 8, 0};
static const int ShelterMg = 9;
static const int TempoBonus = 14;
static const int SpaceMg = 1, SpaceEg = 0;

inline U64 pawn_attacks_bb(U64 pawns, int c) {
    return (c == WHITE) ? (((pawns << 7) & ~FILE_H_BB) | ((pawns << 9) & ~FILE_A_BB))
                        : (((pawns >> 7) & ~FILE_A_BB) | ((pawns >> 9) & ~FILE_H_BB));
}

inline bool has_non_pawn(const Position& p, int c) {
    return (p.bb[make_pc(c, KNIGHT)] | p.bb[make_pc(c, BISHOP)] |
            p.bb[make_pc(c, ROOK)] | p.bb[make_pc(c, QUEEN)]) != 0;
}

int evaluate_classical(const Position& p) {
    int mg = p.mg, eg = p.eg;

    int termMg[2] = {0, 0}, termEg[2] = {0, 0}, danger[2] = {0, 0};
    U64 attN[2], attB[2], attR[2], attQ[2], oppPawnAttC[2], ownPawnAttC[2];
    U64 ownPawnC[2], oppPawnC[2], zone[2];

    for (int c = WHITE; c <= BLACK; ++c) {
        int ks = lsb(p.bb[make_pc(c, KING)]);
        zone[c] = KingAtt[ks] | bit(ks);
    }

    // ---------- pass 1: mobility, rook files, attack maps ----------
    for (int c = WHITE; c <= BLACK; ++c) {
        const int o = c ^ 1;
        ownPawnC[c] = p.bb[make_pc(c, PAWN)];
        oppPawnC[c] = p.bb[make_pc(o, PAWN)];
        ownPawnAttC[c] = pawn_attacks_bb(ownPawnC[c], c);
        oppPawnAttC[c] = pawn_attacks_bb(oppPawnC[c], o);
        const U64 mobilityArea = ~(ownPawnC[c] | oppPawnAttC[c] | p.bb[make_pc(c, KING)]);
        const U64 enemyZone = zone[o];

        int cmg = 0, ceg = 0;
        U64 aN = 0, aB = 0, aR = 0, aQ = 0;

        U64 kn = p.bb[make_pc(c, KNIGHT)];
        while (kn) {
            int s = lsb(kn); kn &= kn - 1;
            U64 a = KnightAtt[s];
            int m = popcount(a & mobilityArea);
            cmg += MobMg[KNIGHT] * m; ceg += MobEg[KNIGHT] * m;
            aN |= a;
            if (a & enemyZone) danger[o] += KingAttWeight[KNIGHT];
        }
        U64 bi = p.bb[make_pc(c, BISHOP)];
        while (bi) {
            int s = lsb(bi); bi &= bi - 1;
            U64 a = bishop_att(s, p.all);
            int m = popcount(a & mobilityArea);
            cmg += MobMg[BISHOP] * m; ceg += MobEg[BISHOP] * m;
            aB |= a;
            if (a & enemyZone) danger[o] += KingAttWeight[BISHOP];
        }
        U64 rk = p.bb[make_pc(c, ROOK)];
        while (rk) {
            int s = lsb(rk); rk &= rk - 1;
            U64 a = rook_att(s, p.all);
            int m = popcount(a & mobilityArea);
            cmg += MobMg[ROOK] * m; ceg += MobEg[ROOK] * m;
            aR |= a;
            if (a & enemyZone) danger[o] += KingAttWeight[ROOK];
            int f = file_of(s);
            if (!(ownPawnC[c] & FileBB[f])) {
                if (!(oppPawnC[c] & FileBB[f])) { cmg += RookOpenMg; ceg += RookOpenEg; }
                else                            { cmg += RookSemiMg; ceg += RookSemiEg; }
            }
            int seventh = (c == WHITE) ? 6 : 1;
            if (rank_of(s) == seventh) { cmg += Rook7thMg; ceg += Rook7thEg; }
        }
        U64 qu = p.bb[make_pc(c, QUEEN)];
        while (qu) {
            int s = lsb(qu); qu &= qu - 1;
            U64 a = bishop_att(s, p.all) | rook_att(s, p.all);
            int m = popcount(a & mobilityArea);
            cmg += MobMg[QUEEN] * m; ceg += MobEg[QUEEN] * m;
            aQ |= a;
            if (a & enemyZone) danger[o] += KingAttWeight[QUEEN];
        }

        danger[o] += 2 * popcount(ownPawnAttC[c] & enemyZone);

        attN[c] = aN; attB[c] = aB; attR[c] = aR; attQ[c] = aQ;
        termMg[c] = cmg; termEg[c] = ceg;
    }

    // ---------- pass 2: pawns, threats, king safety ----------
    for (int c = WHITE; c <= BLACK; ++c) {
        const int sign = (c == WHITE) ? 1 : -1;
        const int o = c ^ 1;
        const U64 ownPawns = ownPawnC[c];
        const U64 oppPawns = oppPawnC[c];
        const U64 ownPawnAtt = ownPawnAttC[c];
        const U64 oppPawnAtt = oppPawnAttC[c];
        int cmg = termMg[c], ceg = termEg[c];

        const U64 attMinor = attN[c] | attB[c];
        const U64 attRook = attR[c];
        const U64 attQueen = attQ[c];

        if (popcount(p.bb[make_pc(c, BISHOP)]) >= 2) { cmg += BishopPairMg; ceg += BishopPairEg; }

        U64 pp = ownPawns;
        while (pp) {
            int s = lsb(pp); pp &= pp - 1;
            int f = file_of(s);
            U64 fwd = (c == WHITE) ? RankAbove[rank_of(s)] : RankBelow[rank_of(s)];
            if (!(PassedMask[c][s] & oppPawns)) {
                int relRank = (c == WHITE) ? rank_of(s) : 7 - rank_of(s);
                cmg += PassedMg[relRank]; ceg += PassedEg[relRank];
            }
            if (!(ownPawns & AdjFilesBB[f])) { cmg += IsolatedMg; ceg += IsolatedEg; }
            if (ownPawns & FileBB[f] & fwd) { cmg += DoubledMg; ceg += DoubledEg; }
            else {
                U64 stopSq = (c == WHITE) ? bit(s + 8) : bit(s - 8);
                if (rank_of(s) > 1 && rank_of(s) < 6 &&
                    !(stopSq & p.all) && (stopSq & oppPawnAtt)) {
                    cmg += BackwardMg; ceg += BackwardEg;
                }
            }
            if (PawnAtt[c ^ 1][s] & ownPawns) { cmg += SupportedMg; ceg += SupportedEg; }
        }

        U64 theirMinor = p.bb[make_pc(o, KNIGHT)] | p.bb[make_pc(o, BISHOP)];
        U64 theirMajor = p.bb[make_pc(o, ROOK)] | p.bb[make_pc(o, QUEEN)];
        int tpm = popcount(ownPawnAtt & theirMinor);
        int tpg = popcount(ownPawnAtt & theirMajor);
        int tmm = popcount(attMinor & theirMajor);
        int loose = popcount((attMinor | attRook | attQueen) & p.occ[o] & ~ownPawns & ~ownPawnAtt);
        cmg += tpm * ThreatPawnMinorMg + tpg * ThreatPawnMajorMg + tmm * ThreatMinorMajorMg +
               loose * ThreatLooseMg;
        ceg += tpm * ThreatPawnMinorEg + tpg * ThreatPawnMajorEg + tmm * ThreatMinorMajorEg +
               loose * ThreatLooseEg;

        int ksq = lsb(p.bb[make_pc(c, KING)]);
        int kf = file_of(ksq), kr = rank_of(ksq);
        cmg -= danger[c] * danger[c] / 6;

        U64 shelterArea = 0;
        for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); ++f)
            for (int d = 1; d <= 2; ++d) {
                int r = (c == WHITE) ? kr + d : kr - d;
                if (r >= 0 && r < 8) shelterArea |= bit(r * 8 + f);
            }
        cmg += (popcount(shelterArea & ownPawns) - 3) * ShelterMg;

        mg += sign * cmg;
        eg += sign * ceg;
    }

    int space = popcount(SpaceBB & ~p.bb[W_P] & ~p.bb[B_P] & ~p.occ[BLACK]) -
                popcount(SpaceBB & ~p.bb[W_P] & ~p.bb[B_P] & ~p.occ[WHITE]);
    mg += space * SpaceMg; eg += space * SpaceEg;

    int phase = p.phaseInc;
    if (phase > 24) phase = 24;
    int v = (mg * phase + eg * (24 - phase)) / 24;
    v += (p.stm == WHITE) ? TempoBonus : -TempoBonus;
    return (p.stm == WHITE) ? v : -v;
}

inline int evaluate(const Position& p) {
    return Nnue::available() ? Nnue::evaluate(p.nnue, p.stm) : evaluate_classical(p);
}

// ---------------------------------------------------------------------------
//  Transposition table
// ---------------------------------------------------------------------------
enum : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    U64 key;
    int16_t value;
    uint16_t move;
    int8_t depth;
    uint8_t genBound;
};
static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

struct TT {
    std::vector<TTEntry> t;
    U64 bucketMask = 0;
    uint8_t gen = 0;

    void resize(int mb) {
        if (mb < 1) mb = 1;
        if (mb > 8192) mb = 8192;
        size_t bytes = (size_t)mb * 1024ULL * 1024ULL;
        size_t buckets = bytes / (sizeof(TTEntry) * 4);
        size_t p2 = 1;
        while (p2 * 2 <= buckets) p2 *= 2;
        t.assign(p2 * 4, TTEntry{});
        bucketMask = (U64)(p2 - 1);
    }
    void clear() { std::fill(t.begin(), t.end(), TTEntry{}); }
    inline TTEntry* bucket(U64 key) { return &t[(key & bucketMask) * 4]; }

    inline bool probe(U64 key, uint16_t& mv, int& score, int& depth, int& bound) {
        if (t.empty()) return false;
        TTEntry* b = bucket(key);
        for (int i = 0; i < 4; ++i) {
            if (b[i].key == key && (b[i].genBound & 3) != BOUND_NONE) {
                mv = b[i].move;
                score = b[i].value;
                depth = b[i].depth;
                bound = b[i].genBound & 3;
                return true;
            }
        }
        mv = b[0].move;
        return false;
    }

    inline void store(U64 key, int score, uint16_t mv, int depth, int bound) {
        if (t.empty()) return;
        TTEntry* b = bucket(key);
        TTEntry* slot = nullptr;
        for (int i = 0; i < 4; ++i) if (b[i].key == key) { slot = &b[i]; break; }
        if (!slot) {
            slot = &b[0];
            for (int i = 1; i < 4; ++i) {
                int si = (slot->genBound >> 2), bi2 = (b[i].genBound >> 2);
                if (bi2 != gen && si == gen) slot = &b[i];
                else if (bi2 == si && b[i].depth < slot->depth) slot = &b[i];
            }
        } else {
            if (slot->depth > depth + 2 && bound != BOUND_EXACT) return;
            if (mv == 0) mv = slot->move;
        }
        slot->key = key;
        slot->value = (int16_t)score;
        slot->move = mv;
        slot->depth = (int8_t)depth;
        slot->genBound = (uint8_t)((gen << 2) | bound);
    }

    int hashfull() {
        if (t.empty()) return 0;
        int cnt = 0;
        size_t lim = std::min<size_t>(t.size(), 4000);
        for (size_t i = 0; i < lim; ++i) if ((t[i].genBound >> 2) == gen) cnt++;
        return (int)((int64_t)cnt * 1000 / (int64_t)std::max<size_t>(lim, 1));
    }
};

TT tt;

inline int score_to_tt(int s, int ply) {
    if (s >= MATE_IN_MAX) return s + ply;
    if (s <= -MATE_IN_MAX) return s - ply;
    return s;
}
inline int score_from_tt(int s, int ply) {
    if (s >= MATE_IN_MAX) return s - ply;
    if (s <= -MATE_IN_MAX) return s + ply;
    return s;
}

// ---------------------------------------------------------------------------
//  Globals / time
// ---------------------------------------------------------------------------
std::atomic<bool> g_stop{false};

struct Limits {
    int64_t soft = 0, hard = 0;
    bool infinite = false;
};
Limits g_lim;

// remembered so that `ponderhit` can switch from pondering to the real clock
struct GoParams {
    int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
    int movestogo = 0, movetime = 0;
    int stm = WHITE;
    bool valid = false;
};
GoParams g_go;
Clock::time_point g_start;

inline int64_t elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_start).count();
}

struct ThreadData {
    Position pos;
    uint16_t killers[MAX_PLY + 2][2];
    uint16_t counter[16][64];
    int16_t history[16][64];
    std::vector<int16_t> contHist;
    uint16_t moveStack[MAX_PLY + 2];
    int evalStack[MAX_PLY + 2];
    U64 nodes = 0;
    int seldepth = 0;
    int id = 0;

    void init() {
        std::memset(killers, 0, sizeof(killers));
        std::memset(counter, 0, sizeof(counter));
        std::memset(history, 0, sizeof(history));
        std::memset(moveStack, 0, sizeof(moveStack));
        for (int i = 0; i < MAX_PLY + 2; ++i) evalStack[i] = VALUE_NONE;
        contHist.assign(16 * 64 * 16 * 64, 0);
    }
    inline int16_t* ch(int prevPc, int prevTo, int pc, int to) {
        return &contHist[(((prevPc * 64) + prevTo) * 16 + pc) * 64 + to];
    }
};

std::vector<ThreadData*> threads;

inline bool check_stop(ThreadData& T) {
    if (g_stop.load(std::memory_order_relaxed)) return true;
    if (T.id != 0) return false;
    if (g_lim.hard > 0 && elapsed_ms() >= g_lim.hard) {
        g_stop.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

inline bool is_repetition(const Position& p) {
    int end = std::max(0, p.histSize - p.fifty - 1);
    for (int i = p.histSize - 2; i >= end; i -= 2)
        if (p.keyHist[i] == p.key) return true;
    return false;
}

struct NullUndo { int ep, fifty; U64 key; };

inline void make_null(Position& p, NullUndo& nu) {
    nu.ep = p.ep; nu.fifty = p.fifty; nu.key = p.key;
    p.key ^= ep_key(p, p.stm, p.ep);
    p.ep = -1;
    p.stm ^= 1;
    p.key ^= Zside;
    p.fifty++;
    if (p.histSize < MAX_HISTORY) p.keyHist[p.histSize] = p.key;
    p.histSize++;
}

inline void unmake_null(Position& p, const NullUndo& nu) {
    p.histSize--;
    p.stm ^= 1;
    p.ep = nu.ep;
    p.fifty = nu.fifty;
    p.key = nu.key;
}

// ---------------------------------------------------------------------------
//  Quiescence
// ---------------------------------------------------------------------------
int qsearch(ThreadData& T, Position& pos, int alpha, int beta, int ply) {
    T.nodes++;
    if ((T.nodes & 1023) == 0 && check_stop(T)) return 0;
    if (ply > T.seldepth) T.seldepth = ply;
    if (ply >= MAX_PLY - 2) return evaluate(pos);
    // a repetition needs at least four plies since the last irreversible move
    if (pos.fifty >= 100 || (pos.fifty >= 4 && is_repetition(pos))) return 0;

    const bool chk = in_check(pos);
    int best;
    if (chk) {
        best = -INF;
    } else {
        best = evaluate(pos);
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    }

    MoveList ml;
    if (chk) gen_all<false>(pos, ml);
    else     gen_all<true>(pos, ml);

    int scores[MAX_MOVES];
    for (int i = 0; i < ml.n; ++i) {
        const uint16_t m = ml.m[i];
        const int mt = mv_type(m);
        int s;
        if (is_capture(mt)) {
            int victim = (mt == MT_EP) ? PAWN : pc_type(pos.sq[mv_to(m)]);
            int attacker = pc_type(pos.sq[mv_from(m)]);
            if (SEE_VAL[victim] < SEE_VAL[attacker]) s = -100000 + victim;
            else                                     s = victim * 64 - attacker;
        } else if (is_promo(mt)) {
            s = 50000 + promo_type(mt);
        } else {
            s = 0;
        }
        scores[i] = s;
    }

    int moveCount = 0;
    for (int i = 0; i < ml.n; ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.n; ++j) if (scores[j] > scores[bi]) bi = j;
        if (bi != i) { std::swap(ml.m[i], ml.m[bi]); std::swap(scores[i], scores[bi]); }

        const uint16_t m = ml.m[i];
        if (!chk && !see_ge(pos, m, 0)) continue;

        const int movedPc = pos.sq[mv_from(m)];
        Undo u;
        make_move(pos, m, u);
        if (illegal(pos)) { unmake(pos, u, movedPc); continue; }
        moveCount++;

        int v = -qsearch(T, pos, -beta, -alpha, ply + 1);
        unmake(pos, u, movedPc);
        if (g_stop.load(std::memory_order_relaxed)) return 0;

        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) return v;
            }
        }
    }

    if (chk && moveCount == 0) return -MATE + ply;
    return best;
}

// ---------------------------------------------------------------------------
//  Main search
// ---------------------------------------------------------------------------
int negamax(ThreadData& T, Position& pos, int depth, int alpha, int beta, int ply, bool cutNode) {
    const bool pv = (beta - alpha) > 1;

    if (depth <= 0) return qsearch(T, pos, alpha, beta, ply);

    T.nodes++;
    if ((T.nodes & 1023) == 0 && check_stop(T)) return 0;
    if (ply > T.seldepth) T.seldepth = ply;
    if (ply >= MAX_PLY - 2) return evaluate(pos);

    const bool chk = in_check(pos);
    if (chk) depth++;

    if (pos.fifty >= 100 || (pos.fifty >= 4 && is_repetition(pos))) return 0;

    if (alpha < -MATE + ply) alpha = -MATE + ply;
    if (beta > MATE - ply - 1) beta = MATE - ply - 1;
    if (alpha >= beta) return alpha;

    uint16_t ttMove = 0;
    int ttScore = 0, ttDepth = 0, ttBound = BOUND_NONE;
    bool ttHit = tt.probe(pos.key, ttMove, ttScore, ttDepth, ttBound);
    if (ttHit && !pv && ttDepth >= depth) {
        int s = score_from_tt(ttScore, ply);
        if (ttBound == BOUND_EXACT ||
            (ttBound == BOUND_LOWER && s >= beta) ||
            (ttBound == BOUND_UPPER && s <= alpha))
            return s;
    }

    if (!pv && !chk && depth >= 4 && !ttMove) depth--;

    int eval = 0;
    if (!chk) eval = evaluate(pos);
    T.evalStack[ply] = chk ? VALUE_NONE : eval;
    const bool improving = ply >= 2 && T.evalStack[ply - 2] != VALUE_NONE &&
                           eval > T.evalStack[ply - 2];

    if (!pv && !chk) {
        int margin = improving ? 65 * depth : 85 * depth;
        if (depth <= 9 && eval - margin >= beta && std::abs(beta) < MATE_IN_MAX)
            return eval;

        if (depth <= 3 && eval + 300 + 200 * depth <= alpha) {
            int v = qsearch(T, pos, alpha, beta, ply);
            if (v <= alpha && !g_stop.load(std::memory_order_relaxed)) return v;
        }
    }

    if (!pv && !chk && depth >= 2 && eval >= beta && has_non_pawn(pos, pos.stm) &&
        std::abs(beta) < MATE_IN_MAX) {
        int R = 3 + depth / 5 + std::min(3, (eval - beta) / 200);
        if (R > depth - 2) R = std::max(1, depth - 2);
        NullUndo nu;
        make_null(pos, nu);
        int v = -negamax(T, pos, depth - R - 1, -beta, -beta + 1, ply + 1, !cutNode);
        unmake_null(pos, nu);
        if (g_stop.load(std::memory_order_relaxed)) return 0;
        if (v >= beta) return (v >= MATE_IN_MAX) ? beta : v;
    }

    if (!pv && !chk && depth >= 5 && std::abs(beta) < MATE_IN_MAX) {
        const int pcBeta = beta + 170;
        MoveList caps;
        gen_all<true>(pos, caps);
        int cs[MAX_MOVES];
        for (int i = 0; i < caps.n; ++i) {
            const uint16_t m = caps.m[i];
            const int mt = mv_type(m);
            int victim = (mt == MT_EP) ? PAWN : pc_type(pos.sq[mv_to(m)]);
            int attacker = pc_type(pos.sq[mv_from(m)]);
            cs[i] = victim * 64 - attacker + (is_promo(mt) ? 5000 : 0);
        }
        for (int i = 0; i < caps.n; ++i) {
            int bi = i;
            for (int j = i + 1; j < caps.n; ++j) if (cs[j] > cs[bi]) bi = j;
            if (bi != i) { std::swap(caps.m[i], caps.m[bi]); std::swap(cs[i], cs[bi]); }

            const uint16_t m = caps.m[i];
            if (!see_ge(pos, m, pcBeta - beta + 100)) continue;
            const int movedPc = pos.sq[mv_from(m)];
            Undo u;
            make_move(pos, m, u);
            if (illegal(pos)) { unmake(pos, u, movedPc); continue; }
            int v = -negamax(T, pos, depth - 4, -pcBeta, -pcBeta + 1, ply + 1, !cutNode);
            unmake(pos, u, movedPc);
            if (g_stop.load(std::memory_order_relaxed)) return 0;
            if (v >= pcBeta) return v;
        }
    }

    MoveList ml;
    gen_all<false>(pos, ml);
    if (ml.n == 0) return chk ? -MATE + ply : 0;

    const int prevTo = (ply > 0 && T.moveStack[ply - 1]) ? mv_to(T.moveStack[ply - 1]) : 0;
    const int prevPc = (ply > 0 && T.moveStack[ply - 1]) ? (int)pos.sq[prevTo] : 0;
    const uint16_t counterMove = T.counter[prevPc & 15][prevTo & 63];

    int scores[MAX_MOVES];
    for (int i = 0; i < ml.n; ++i) {
        const uint16_t m = ml.m[i];
        const int mt = mv_type(m);
        const int to = mv_to(m);
        const int pc = pos.sq[mv_from(m)];
        if (m == ttMove) { scores[i] = 1 << 30; continue; }
        if (is_promo(mt)) { scores[i] = (1 << 28) + promo_type(mt); continue; }
        if (is_capture(mt)) {
            int victim = (mt == MT_EP) ? PAWN : pc_type(pos.sq[to]);
            int attacker = pc_type(pc);
            if (SEE_VAL[victim] < SEE_VAL[attacker]) scores[i] = -9000 + victim;
            else                                    scores[i] = (1 << 27) + victim * 64 - attacker;
            continue;
        }
        if (m == T.killers[ply][0]) { scores[i] = (1 << 26); continue; }
        if (m == T.killers[ply][1]) { scores[i] = (1 << 26) - 1; continue; }
        if (m == counterMove)       { scores[i] = (1 << 25); continue; }
        scores[i] = T.history[pc][to] + *T.ch(prevPc, prevTo, pc, to);
    }

    const int origAlpha = alpha;
    int bestScore = -INF;
    uint16_t bestMove = 0;
    int moveCount = 0;
    bool skipQuiets = false;
    uint16_t quietTried[64];
    int nQuiet = 0;

    const int lmpLimit = (3 + depth * depth) / (cutNode ? 1 : 2);
    const bool futilityOk = depth <= 9 && std::abs(alpha) < MATE_IN_MAX;

    for (int i = 0; i < ml.n; ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.n; ++j) if (scores[j] > scores[bi]) bi = j;
        if (bi != i) { std::swap(ml.m[i], ml.m[bi]); std::swap(scores[i], scores[bi]); }

        const uint16_t m = ml.m[i];
        const int mt = mv_type(m);
        const bool cap = is_capture(mt);
        const bool promo = is_promo(mt);

        if (skipQuiets && !cap && !promo) continue;

        if (!chk && !pv && moveCount > 0 && bestScore > -MATE_IN_MAX) {
            if (!cap && !promo) {
                if (moveCount >= lmpLimit) { skipQuiets = true; continue; }
                if (futilityOk && eval + 110 + 95 * depth <= alpha) { skipQuiets = true; continue; }
            } else if (depth <= 6 && !see_ge(pos, m, -110)) {
                continue;
            }
        }

        const int movedPc = pos.sq[mv_from(m)];
        const int newDepth = depth - 1;

        Undo u;
        make_move(pos, m, u);
        if (illegal(pos)) { unmake(pos, u, movedPc); continue; }

        T.moveStack[ply] = m;
        moveCount++;

        int score;
        if (moveCount == 1) {
            score = -negamax(T, pos, newDepth, -beta, -alpha, ply + 1, false);
        } else {
            int r = 0;
            if (!cap && !promo) {
                int h = T.history[movedPc][mv_to(m)] + *T.ch(prevPc, prevTo, movedPc, mv_to(m));
                if (depth >= 3 && moveCount >= 3) {
                    r = LMRTable[std::min(depth, 63)][std::min(moveCount, 63)];
                    r -= std::min(2, std::max(-1, h / 8192));
                    if (pv) r--;
                    else if (cutNode) r++;
                    if (!improving) r++;
                    r = std::max(0, std::min(r, newDepth - 1));
                }
                if (nQuiet < 64) quietTried[nQuiet++] = m;
            }
            score = -negamax(T, pos, newDepth - r, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && r > 0)
                score = -negamax(T, pos, newDepth, -alpha - 1, -alpha, ply + 1, !cutNode);
            if (score > alpha && score < beta)
                score = -negamax(T, pos, newDepth, -beta, -alpha, ply + 1, false);
        }

        unmake(pos, u, movedPc);
        if (g_stop.load(std::memory_order_relaxed)) return 0;

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) {
                    if (!cap && !promo) {
                        if (T.killers[ply][0] != m) {
                            T.killers[ply][1] = T.killers[ply][0];
                            T.killers[ply][0] = m;
                        }
                        int bonus = std::min(1600, depth * depth * 24);
                        hist_bonus(T.history[movedPc][mv_to(m)], bonus);
                        hist_bonus(*T.ch(prevPc, prevTo, movedPc, mv_to(m)), bonus);
                        T.counter[prevPc & 15][prevTo & 63] = m;
                        for (int k = 0; k < nQuiet; ++k) {
                            uint16_t q = quietTried[k];
                            if (q == m) continue;
                            int qpc = pos.sq[mv_from(q)];
                            hist_malus(T.history[qpc][mv_to(q)], bonus);
                            hist_malus(*T.ch(prevPc, prevTo, qpc, mv_to(q)), bonus);
                        }
                    }
                    break;
                }
            }
        }
    }

    // every pseudo-legal move was illegal: only possible when in check
    if (bestMove == 0) return chk ? -MATE + ply : 0;

    uint8_t bound = (bestScore >= beta) ? BOUND_LOWER
                  : (bestScore > origAlpha ? BOUND_EXACT : BOUND_UPPER);
    if (!g_stop.load(std::memory_order_relaxed))
        tt.store(pos.key, score_to_tt(bestScore, ply), bestMove, depth, bound);
    return bestScore;
}

// ---------------------------------------------------------------------------
//  Move printing / parsing
// ---------------------------------------------------------------------------
std::string move_str(const Position&, uint16_t m) {
    if (m == 0) return "0000";
    std::string s;
    s += char('a' + file_of(mv_from(m)));
    s += char('1' + rank_of(mv_from(m)));
    s += char('a' + file_of(mv_to(m)));
    s += char('1' + rank_of(mv_to(m)));
    if (is_promo(mv_type(m))) {
        const char pc[] = {' ', 'n', 'b', 'r', 'q'};
        s += pc[promo_type(mv_type(m))];
    }
    return s;
}

std::string score_string(int s) {
    if (std::abs(s) >= MATE_IN_MAX) {
        int n = (MATE - std::abs(s) + 1) / 2;
        return "mate " + std::to_string(s > 0 ? n : -n);
    }
    return "cp " + std::to_string(s);
}

bool parse_uci_move(Position& p, const std::string& s, uint16_t& out) {
    if (s.size() < 4) return false;
    int from = (s[1] - '1') * 8 + (s[0] - 'a');
    int to   = (s[3] - '1') * 8 + (s[2] - 'a');
    int promo = -1;
    if (s.size() >= 5) {
        switch (s[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK;   break;
            case 'q': promo = QUEEN;  break;
            default: break;
        }
    }
    MoveList ml;
    gen_all<false>(p, ml);
    for (int i = 0; i < ml.n; ++i) {
        const uint16_t m = ml.m[i];
        if (mv_from(m) != from || mv_to(m) != to) continue;
        if (promo >= 0) {
            if (is_promo(mv_type(m)) && promo_type(mv_type(m)) == promo) { out = m; return true; }
        } else if (!is_promo(mv_type(m))) { out = m; return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
//  Root search
// ---------------------------------------------------------------------------
struct RootMove {
    uint16_t m;
    int score;
};

void iterate(ThreadData& T, int maxDepth, bool mainThread) {
    Position& pos = T.pos;

    MoveList ml;
    gen_all<false>(pos, ml);
    std::vector<RootMove> rm;
    for (int i = 0; i < ml.n; ++i) {
        const int movedPc = pos.sq[mv_from(ml.m[i])];
        Undo u;
        make_move(pos, ml.m[i], u, false);
        if (!illegal(pos)) rm.push_back({ml.m[i], -INF});
        unmake(pos, u, movedPc, false);
    }
    if (rm.empty()) {
        if (mainThread) std::cout << "bestmove 0000" << std::endl;
        return;
    }

    uint16_t bestMove = rm[0].m;
    int prevScore = 0;
    int delta = 16;
    T.nodes = 0;          // cumulative for this iterate() call

    for (int d = 1; d <= maxDepth; ++d) {
        if (mainThread && d > 1 && g_lim.soft > 0 && elapsed_ms() >= g_lim.soft) break;

        T.seldepth = 0;

        int alpha, beta;
        if (d == 1) { alpha = -INF; beta = INF; }
        else {
            alpha = std::max(-INF, prevScore - delta);
            beta  = std::min(INF, prevScore + delta);
        }

        int localBest = -INF;
        while (true) {
            localBest = -INF;
            int a = alpha;
            for (size_t i = 0; i < rm.size(); ++i) {
                const int movedPc = pos.sq[mv_from(rm[i].m)];
                Undo u;
                make_move(pos, rm[i].m, u);
                int sc;
                if (i == 0) {
                    sc = -negamax(T, pos, d - 1, -beta, -a, 1, false);
                } else {
                    sc = -negamax(T, pos, d - 1, -a - 1, -a, 1, true);
                    if (sc > a) sc = -negamax(T, pos, d - 1, -beta, -a, 1, false);
                }
                unmake(pos, u, movedPc);
                if (g_stop.load(std::memory_order_relaxed)) break;
                rm[i].score = sc;
                if (sc > localBest) localBest = sc;
                if (sc > a) a = sc;
            }
            if (g_stop.load(std::memory_order_relaxed)) break;

            if (localBest <= alpha && alpha > -INF) {
                beta = (alpha + beta) / 2;
                alpha = std::max(-INF, localBest - delta);
                delta += delta / 2 + 4;
                continue;
            }
            if (localBest >= beta && beta < INF) {
                beta = std::min(INF, localBest + delta);
                delta += delta / 2 + 4;
                continue;
            }
            break;
        }
        if (g_stop.load(std::memory_order_relaxed)) break;

        std::stable_sort(rm.begin(), rm.end(),
                         [](const RootMove& x, const RootMove& y) { return x.score > y.score; });
        bestMove = rm[0].m;
        prevScore = rm[0].score;
        delta = 16;

        if (mainThread) {
            int64_t ms = elapsed_ms();
            U64 nps = ms ? (T.nodes * 1000ULL / (U64)ms) : 0;
            std::cout << "info depth " << d
                      << " seldepth " << T.seldepth
                      << " score " << score_string(prevScore)
                      << " nodes " << T.nodes
                      << " nps " << nps
                      << " hashfull " << tt.hashfull()
                      << " time " << ms
                      << " pv " << move_str(pos, bestMove)
                      << std::endl;
        }

        if (std::abs(prevScore) > MATE_IN_MAX) break;
        if (mainThread && g_lim.soft > 0 && elapsed_ms() >= g_lim.soft) break;
    }

    if (mainThread) std::cout << "bestmove " << move_str(pos, bestMove) << std::endl;
}

void helper_main(ThreadData* T, int maxDepth) {
    int phase = T->id & 1;
    for (int d = 1; d <= maxDepth; ++d) {
        if (g_stop.load(std::memory_order_relaxed)) break;
        iterate(*T, std::min(maxDepth, d + phase), false);
    }
}

// ---------------------------------------------------------------------------
//  perft
// ---------------------------------------------------------------------------
U64 perft(Position& p, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    gen_all<false>(p, ml);
    U64 n = 0;
    for (int i = 0; i < ml.n; ++i) {
        const int movedPc = p.sq[mv_from(ml.m[i])];
        Undo u;
        make_move(p, ml.m[i], u, false);
        if (!illegal(p)) n += perft(p, depth - 1);
        unmake(p, u, movedPc, false);
    }
    return n;
}

// ---------------------------------------------------------------------------
//  FEN
// ---------------------------------------------------------------------------
bool set_fen(Position& p, const std::string& fen) {
    std::memset(p.bb, 0, sizeof(p.bb));
    std::memset(p.byType, 0, sizeof(p.byType));
    p.occ[0] = p.occ[1] = 0;
    p.all = 0;
    std::memset(p.sq, 0, sizeof(p.sq));
    p.stm = WHITE; p.castling = 0; p.ep = -1; p.fifty = 0;
    p.mg = p.eg = p.phaseInc = 0; p.key = 0; p.histSize = 0;
    p.nnue = Nnue::Accumulator{};

    std::istringstream ss(fen);
    std::string placement, stm, castle, ep;
    if (!(ss >> placement >> stm >> castle >> ep)) return false;

    int rank = 7, file = 0;
    for (char ch : placement) {
        if (ch == '/') { rank--; file = 0; continue; }
        if (ch >= '1' && ch <= '8') { file += ch - '0'; continue; }
        if (rank < 0 || rank > 7 || file < 0 || file > 7) return false;
        int color = std::isupper((unsigned char)ch) ? WHITE : BLACK;
        int type;
        switch (std::tolower((unsigned char)ch)) {
            case 'p': type = PAWN; break;
            case 'n': type = KNIGHT; break;
            case 'b': type = BISHOP; break;
            case 'r': type = ROOK; break;
            case 'q': type = QUEEN; break;
            case 'k': type = KING; break;
            default: return false;
        }
        add_piece(p, make_pc(color, type), rank * 8 + file);
        file++;
    }

    p.stm = (stm == "w") ? WHITE : BLACK;
    for (char ch : castle) {
        switch (ch) {
            case 'K': p.castling |= 1; break;
            case 'Q': p.castling |= 2; break;
            case 'k': p.castling |= 4; break;
            case 'q': p.castling |= 8; break;
            default: break;
        }
    }
    if (ep != "-" && ep.size() >= 2)
        p.ep = (ep[1] - '1') * 8 + (ep[0] - 'a');

    int hm = 0;
    if (ss >> hm) p.fifty = hm;

    // recompute key so that incremental and computed hashes agree
    p.key ^= p.stm == BLACK ? Zside : 0;
    p.key ^= Zcastle[p.castling];
    p.key ^= ep_key(p, p.stm, p.ep);

    // a position without both kings would break evaluation and legality tests
    if (!p.bb[W_K] || !p.bb[B_K]) return false;

    Nnue::reset(p.nnue, p.sq);
    p.keyHist[0] = p.key;
    p.histSize = 1;
    return true;
}

// Recomputes every incremental quantity from scratch (debug / self-test).
void recompute(const Position& p, int& mg, int& eg, int& ph, U64& key) {
    mg = eg = ph = 0;
    key = 0;
    for (int pc = 1; pc <= 14; ++pc) {
        if (pc == 7 || pc == 8) continue;
        U64 b = p.bb[pc];
        while (b) {
            int s = lsb(b); b &= b - 1;
            int sgn = (pc_color(pc) == WHITE) ? 1 : -1;
            mg += sgn * PSQ_MG[pc][s];
            eg += sgn * PSQ_EG[pc][s];
            ph += PhaseInc[pc];
            key ^= Zpiece[pc][s];
        }
    }
    if (p.stm == BLACK) key ^= Zside;
    key ^= Zcastle[p.castling];
    key ^= ep_key(p, p.stm, p.ep);
}

U64 compute_key(Position& p) {
    int mg, eg, ph;
    U64 k;
    recompute(p, mg, eg, ph, k);
    return k;
}

bool check_consistency(const Position& p, std::string& why) {
    int mg, eg, ph;
    U64 key;
    recompute(p, mg, eg, ph, key);
    if (key != p.key) { why = "key"; return false; }
    if (mg != p.mg)   { why = "mg"; return false; }
    if (eg != p.eg)   { why = "eg"; return false; }
    if (ph != p.phaseInc) { why = "phase"; return false; }
    // board map vs bitboards
    U64 all = 0;
    for (int s = 0; s < 64; ++s) {
        int pc = p.sq[s];
        if (pc) {
            if (!(p.bb[pc] & bit(s))) { why = "sq/bb"; return false; }
            all |= bit(s);
        }
    }
    if (all != p.all) { why = "all"; return false; }
    if ((p.occ[WHITE] | p.occ[BLACK]) != p.all) { why = "occ"; return false; }
    if (!Nnue::consistent(p.nnue, p.sq)) { why = "nnue"; return false; }
    return true;
}

U64 verify_tree(Position& p, int depth, int& bad) {
    std::string why;
    if (!check_consistency(p, why)) {
        if (!bad) std::cout << "info string INCONSISTENT (" << why << ")" << std::endl;
        bad++;
        return 0;
    }
    if (depth == 0) return 1;
    MoveList ml;
    gen_all<false>(p, ml);
    U64 n = 0;
    for (int i = 0; i < ml.n; ++i) {
        const int movedPc = p.sq[mv_from(ml.m[i])];
        Undo u;
        make_move(p, ml.m[i], u);
        if (!illegal(p)) n += verify_tree(p, depth - 1, bad);
        unmake(p, u, movedPc);
    }
    return n;
}

// ---------------------------------------------------------------------------
//  UCI
// ---------------------------------------------------------------------------
std::thread searchThread;
std::atomic<bool> g_searching{false};
int g_threadCount = 1;
int g_hashMB = 256;
int g_moveOverhead = 30;

void stop_search() {
    g_stop.store(true, std::memory_order_relaxed);
    if (searchThread.joinable()) searchThread.join();
    g_searching.store(false);
}

void apply_go_limits(const GoParams& gp, bool infinite) {
    g_lim.soft = 0;
    g_lim.hard = 0;
    g_lim.infinite = infinite;

    if (gp.movetime > 0) {
        g_lim.hard = std::max<int64_t>(1, gp.movetime - g_moveOverhead);
        g_lim.soft = std::max<int64_t>(1, g_lim.hard * 6 / 10);
    } else if (!infinite && (gp.wtime > 0 || gp.btime > 0)) {
        int64_t myTime = (gp.stm == WHITE) ? gp.wtime : gp.btime;
        int64_t myInc  = (gp.stm == WHITE) ? gp.winc : gp.binc;
        int mtg = gp.movestogo > 0 ? gp.movestogo : 30;
        int64_t usable = std::max<int64_t>(1, myTime - g_moveOverhead - 20);
        int64_t opt = usable / mtg + myInc * 3 / 4;
        opt = std::min<int64_t>(opt, usable / 2);
        opt = std::max<int64_t>(opt, 1);
        int64_t hard = std::min<int64_t>(usable, opt * 3);
        g_lim.soft = opt;
        g_lim.hard = std::max<int64_t>(hard, opt + 5);
        if (myTime < 60) { g_lim.soft = 1; g_lim.hard = std::max<int64_t>(1, myTime / 2); }
    }
}

void init_threads(int n, const Position* seed = nullptr) {
    Position keep;
    if (seed) keep = *seed;
    else if (!threads.empty()) keep = threads[0]->pos;

    for (auto* t : threads) delete t;
    threads.clear();
    for (int i = 0; i < n; ++i) {
        ThreadData* T = new ThreadData();
        T->init();
        T->id = i;
        T->pos = keep;
        threads.push_back(T);
    }
}

void start_search(int maxDepth) {
    if (g_searching.load()) stop_search();
    g_stop.store(false, std::memory_order_relaxed);
    g_start = Clock::now();
    if (tt.gen < 63) tt.gen++; else { tt.gen = 1; }
    for (auto* T : threads) T->init();

    if ((int)threads.size() != g_threadCount) {
        Position seed = threads[0]->pos;
        init_threads(g_threadCount, &seed);
    }
    if (threads.size() > 1) {
        for (size_t i = 1; i < threads.size(); ++i) threads[i]->pos = threads[0]->pos;
    }

    g_searching.store(true);
    searchThread = std::thread([maxDepth]() {
        std::vector<std::thread> helpers;
        for (int i = 1; i < (int)threads.size(); ++i)
            helpers.emplace_back(helper_main, threads[i], maxDepth);
        iterate(*threads[0], maxDepth, true);
        g_stop.store(true, std::memory_order_relaxed);
        for (auto& h : helpers) h.join();
    });
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    init_all();

    std::string evalFile = "nnue/fingolfin.nnue";
    bool loaded = Nnue::load(evalFile);
    if (!loaded && argc > 0) {
        const auto exeDir = std::filesystem::absolute(argv[0]).parent_path();
        evalFile = (exeDir / "nnue" / "fingolfin.nnue").string();
        loaded = Nnue::load(evalFile);
    }
#if defined(WALLE_EMBEDDED_NET)
    // Android package: the network is part of the binary, since a GUI runs the engine
    // from the APK's lib directory where no nnue/ folder exists.
    if (!loaded) {
        loaded = Nnue::load_embedded();
        if (loaded) evalFile = "<embedded network>";
    }
#endif
    if (loaded) std::cerr << "info string NNUE loaded: " << evalFile << '\n';
    else std::cerr << "info string NNUE unavailable; using handcrafted evaluation\n";

    tt.resize(g_hashMB);
    init_threads(g_threadCount);

    Position& board = threads[0]->pos;
    set_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) continue;

        if (token == "uci") {
            std::cout << "id name Wall-E 2\n";
            std::cout << "id author You\n";
            std::cout << "option name Hash type spin default 256 min 1 max 8192\n";
            std::cout << "option name Threads type spin default 1 min 1 max 128\n";
            std::cout << "option name Move Overhead type spin default 30 min 0 max 5000\n";
            std::cout << "option name EvalFile type string default nnue/fingolfin.nnue\n";
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (token == "setoption") {
            std::string w, name, val;
            iss >> w;  // name
            while (iss >> w && w != "value") { if (!name.empty()) name += ' '; name += w; }
            std::getline(iss, val);
            if (!val.empty() && val[0] == ' ') val.erase(0, 1);
            if (name == "Hash") { g_hashMB = std::atoi(val.c_str()); tt.resize(g_hashMB); }
            else if (name == "Threads") { g_threadCount = std::max(1, std::atoi(val.c_str())); }
            else if (name == "Move Overhead") { g_moveOverhead = std::max(0, std::atoi(val.c_str())); }
            else if (name == "EvalFile") {
                stop_search();
                evalFile = val;
                if (!Nnue::load(evalFile)) std::cerr << "info string NNUE load failed; retaining current evaluator: " << evalFile << '\n';
                else {
                    std::cerr << "info string NNUE loaded: " << evalFile << '\n';
                    Nnue::reset(board.nnue, board.sq);
                }
            }
        }
        else if (token == "ucinewgame") {
            stop_search();
            tt.clear();
            for (auto* T : threads) T->init();
        }
        else if (token == "position") {
            stop_search();
            std::string sub;
            iss >> sub;
            bool haveMoves = false;
            if (sub == "startpos") {
                set_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                std::string t;
                if (iss >> t) haveMoves = (t == "moves");
            } else if (sub == "fen") {
                std::string fen, part;
                while (iss >> part) {
                    if (part == "moves") { haveMoves = true; break; }
                    if (!fen.empty()) fen += ' ';
                    fen += part;
                }
                if (!set_fen(board, fen)) {
                    std::cout << "info string bad fen, using startpos: " << fen << std::endl;
                    set_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                    haveMoves = false;
                }
            }
            if (haveMoves) {
                std::string mv;
                while (iss >> mv) {
                    uint16_t m;
                    if (!parse_uci_move(board, mv, m)) {
                        std::cout << "info string rejected move: " << mv << std::endl;
                        break;
                    }
                    Undo u;
                    make_move(board, m, u);
                }
            }
        }
        else if (token == "go") {
            int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
            int movestogo = 0, movetime = 0, maxDepth = 64;
            bool infinite = false;
            std::string w;
            while (iss >> w) {
                if      (w == "wtime")     iss >> wtime;
                else if (w == "btime")     iss >> btime;
                else if (w == "winc")      iss >> winc;
                else if (w == "binc")      iss >> binc;
                else if (w == "movestogo") iss >> movestogo;
                else if (w == "movetime")  iss >> movetime;
                else if (w == "depth")     iss >> maxDepth;
                else if (w == "infinite")  infinite = true;
                else if (w == "nodes")     { int64_t dummy; iss >> dummy; }
                else if (w == "ponder")    infinite = true;
            }

            g_go = GoParams{};
            g_go.wtime = wtime; g_go.btime = btime;
            g_go.winc = winc;   g_go.binc = binc;
            g_go.movestogo = movestogo; g_go.movetime = movetime;
            g_go.stm = board.stm;
            g_go.valid = true;
            apply_go_limits(g_go, infinite);

            // refresh the working copy of the position for every thread
            for (size_t i = 0; i < threads.size(); ++i) {
                if (i == 0) continue;
                threads[i]->pos = threads[0]->pos;
            }
            start_search(maxDepth);
        }
        else if (token == "stop") {
            stop_search();
        }
        else if (token == "ponderhit") {
            // stop pondering and finish the move on the real clock
            if (g_searching.load()) {
                if (g_go.valid) apply_go_limits(g_go, false);
                if (g_lim.hard == 0) { g_lim.soft = 1000; g_lim.hard = 3000; }
            }
        }
        else if (token == "quit") {
            stop_search();
            break;
        }
        else if (token == "d") {
            std::cout << "key " << std::hex << board.key << std::dec
                      << " computed " << compute_key(board) << std::endl;
            std::cout << "fen-ish: stm=" << (board.stm == WHITE ? "w" : "b")
                      << " castling=" << board.castling << " ep=" << board.ep
                      << " fifty=" << board.fifty << std::endl;
            for (int r = 7; r >= 0; --r) {
                std::cout << (r + 1) << " ";
                for (int f = 0; f < 8; ++f) {
                    int pc = board.sq[r * 8 + f];
                    char c = '.';
                    if (pc) {
                        const char sym[] = "?PNBRQK";
                        c = sym[pc_type(pc)];
                        if (pc_color(pc) == BLACK) c = (char)std::tolower(c);
                    }
                    std::cout << c << " ";
                }
                std::cout << "\n";
            }
            std::cout << "  a b c d e f g h" << std::endl;
        }
        else if (token == "perft") {
            int d = 1;
            iss >> d;
            auto t0 = Clock::now();
            for (int i = 1; i <= d; ++i) {
                Position tmp = board;
                U64 n = perft(tmp, i);
                int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 Clock::now() - t0).count();
                std::cout << "info string perft(" << i << ") = " << n
                          << " (" << ms << " ms)" << std::endl;
            }
        }
        else if (token == "eval") {
            std::cout << "info string eval " << evaluate(board) << std::endl;
        }
        else if (token == "verify") {
            int d = 3;
            iss >> d;
            int bad = 0;
            Position tmp = board;
            U64 n = verify_tree(tmp, d, bad);
            std::cout << "info string verify depth " << d << " nodes " << n
                      << " inconsistent " << bad << std::endl;
        }
        else if (token == "bench") {
            static const char* benchFens[] = {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            };
            int depth = 12;
            iss >> depth;
            stop_search();
            Position saved = board;
            U64 total = 0;
            auto t0 = Clock::now();
            for (const char* f : benchFens) {
                set_fen(threads[0]->pos, f);
                g_stop.store(false, std::memory_order_relaxed);
                g_lim.soft = g_lim.hard = 0;
                g_start = Clock::now();
                tt.gen = (tt.gen < 63) ? tt.gen + 1 : 1;
                iterate(*threads[0], depth, false);
                total += threads[0]->nodes;
            }
            int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
            std::cout << "info string bench nodes " << total << " time " << ms
                      << " nps " << (ms ? total * 1000 / (U64)ms : 0) << std::endl;
            board = saved;
        }
        else if (token.empty()) {
            continue;
        }
    }

    stop_search();
    for (auto* T : threads) delete T;
    return 0;
}
