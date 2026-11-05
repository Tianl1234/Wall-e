#include "nnue.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
// The Android x86_64 package is compiled for the SSE4.2 baseline, so the
// compile-time AVX2 kernels below are skipped and the scalar loops would run.
// Give those builds AVX2 versions selected at run time instead: measured on the
// same position set, the scalar loops are ~7x slower than the AVX2 kernels.
#if !defined(__AVX2__)
#define WALLE_X86_AVX2_DISPATCH 1
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define WALLE_NEON 1
#endif
#include <memory>
#include <string>
#include <vector>

namespace Nnue {
namespace {
constexpr int QA = 255;
constexpr int SCALE = 400;
constexpr std::size_t NETWORK_BYTES =
    (std::size_t(INPUT_SIZE) * HIDDEN_SIZE + HIDDEN_SIZE * 3 + 1) * sizeof(int16_t);

struct alignas(64) Network {
    alignas(64) int16_t featureWeights[INPUT_SIZE][HIDDEN_SIZE];
    alignas(64) int16_t bias[HIDDEN_SIZE];
    alignas(64) int16_t outputWeights[2][HIDDEN_SIZE];
    int16_t outputBias;
    // Half of the network's measured side-to-move bonus; subtracted by evaluate().
    int tempoCorrection = 0;
};
std::unique_ptr<Network> owned[NUM_NETS];
// The network the evaluation actually reads; points into owned[].
Network* network = nullptr;

bool network_sane();

// Only the Android package defines WALLE_EMBEDDED_NET. The network is appended to
// the engine binary there (see nnue_embedded.cpp) because a GUI executes the engine
// out of the APK's lib directory, where there is no working directory to hold
// nnue/fingolfin.nnue.
#if defined(WALLE_EMBEDDED_NET)
extern "C" const unsigned char walle_nnue_start[];
extern "C" const unsigned char walle_nnue_end[];
#if defined(WALLE_PHASE_NETS)
extern "C" const unsigned char walle_nnue_endgame_start[];
extern "C" const unsigned char walle_nnue_endgame_end[];
extern "C" const unsigned char walle_nnue_aggressive_start[];
extern "C" const unsigned char walle_nnue_aggressive_end[];
#endif
#endif

bool load_network_memory(const void* data, std::size_t size, int slot) {
    if (data == nullptr || size < NETWORK_BYTES || slot < 0 || slot >= NUM_NETS) return false;
    std::vector<int16_t> raw(NETWORK_BYTES / sizeof(int16_t));
    std::memcpy(raw.data(), data, NETWORK_BYTES);

    auto candidate = std::make_unique<Network>();
    std::size_t offset = 0;
    std::copy_n(raw.data() + offset, std::size_t(INPUT_SIZE) * HIDDEN_SIZE,
                &candidate->featureWeights[0][0]);
    offset += std::size_t(INPUT_SIZE) * HIDDEN_SIZE;
    std::copy_n(raw.data() + offset, HIDDEN_SIZE, candidate->bias);
    offset += HIDDEN_SIZE;
    std::copy_n(raw.data() + offset, HIDDEN_SIZE * 2, &candidate->outputWeights[0][0]);
    offset += HIDDEN_SIZE * 2;
    candidate->outputBias = raw[offset];

    // The SIMD inference path multiplies clipped activations (0..255) by signed
    // weights in 16-bit lanes before widening, so the products must stay inside
    // int16_t.
    for (const auto& perspective : candidate->outputWeights)
        for (int16_t weight : perspective)
            if (weight < -128 || weight > 128) return false;

    // A network whose material ordering is nonsense, or that does not match this
    // code's feature layout, would silently ruin every evaluation and make a
    // GUI's evaluation bar meaningless (it moves the wrong way, or jumps around
    // for equal positions). Reject it so the caller keeps the handcrafted
    // evaluation instead of a broken one.
    Network* previous = network;
    network = candidate.get();
    if (!network_sane()) {
        network = previous;
        std::cerr << "info string NNUE: network fails material sanity checks; rejected\n";
        return false;
    }
    owned[slot] = std::move(candidate);
    network = owned[slot].get();
    return true;
}

bool load_network(const std::string& path, int slot) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::vector<int16_t> raw(NETWORK_BYTES / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(raw.data()), NETWORK_BYTES);
    if (file.gcount() != static_cast<std::streamsize>(NETWORK_BYTES)) return false;
    return load_network_memory(raw.data(), NETWORK_BYTES, slot);
}

inline int feature_index(int piece, int square, int perspective) {
    const int color = piece >> 3;
    const int type = piece & 7;
    const int orientedSquare = square ^ (perspective ? 56 : 0);
    return ((color ^ perspective) * 6 + type - 1) * 64 + orientedSquare;
}

#if defined(WALLE_X86_AVX2_DISPATCH)
// Runtime-selected AVX2 kernels for x86 builds compiled at the SSE4.2 baseline.
// The target attribute lets the intrinsics compile in this translation unit even
// though the rest of the binary stays baseline-compatible.
bool have_avx2() {
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
    }();
    return available;
}

__attribute__((target("avx2")))
void update_row_avx2(int16_t* dst, const int16_t* row, bool add) {
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i weights = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
        value = add ? _mm256_adds_epi16(value, weights) : _mm256_subs_epi16(value, weights);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), value);
    }
}

template <int NA, int NS>
__attribute__((target("avx2")))
void update_rows_avx2(int16_t* dstA, int16_t* dstB,
                      const int16_t* srcA, const int16_t* srcB,
                      const int16_t* const* addA, const int16_t* const* subA,
                      const int16_t* const* addB, const int16_t* const* subB) {
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i vA = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcA + i));
        __m256i vB = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcB + i));
        for (int a = 0; a < NA; ++a) {
            vA = _mm256_adds_epi16(vA, _mm256_load_si256(reinterpret_cast<const __m256i*>(addA[a] + i)));
            vB = _mm256_adds_epi16(vB, _mm256_load_si256(reinterpret_cast<const __m256i*>(addB[a] + i)));
        }
        for (int s = 0; s < NS; ++s) {
            vA = _mm256_subs_epi16(vA, _mm256_load_si256(reinterpret_cast<const __m256i*>(subA[s] + i)));
            vB = _mm256_subs_epi16(vB, _mm256_load_si256(reinterpret_cast<const __m256i*>(subB[s] + i)));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstA + i), vA);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstB + i), vB);
    }
}

__attribute__((target("avx2")))
int infer_avx2(const Accumulator& acc, int stm) {
    const int16_t* us = acc.values[stm];
    const int16_t* them = acc.values[stm ^ 1];
    int64_t total = static_cast<int64_t>(network->outputBias) * QA;
    // Eight independent 32-bit accumulators: enough ILP to hide the multiply
    // and multiply-add latency, and each lane sees at most 32 products of
    // 255 x 255 x 128, far from the int32 limit. (Patricia's MIT design.)
    __m256i sum[8] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256()};
    const __m256i zero = _mm256_setzero_si256();
    const __m256i maxValue = _mm256_set1_epi16(QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 64) {
        for (int j = 0; j < 4; ++j) {
            const int k = i + 16 * j;
            __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(us + k));
            __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(them + k));
            a = _mm256_min_epi16(_mm256_max_epi16(a, zero), maxValue);
            b = _mm256_min_epi16(_mm256_max_epi16(b, zero), maxValue);
            const __m256i wa = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[0] + k));
            const __m256i wb = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[1] + k));
            sum[2 * j] = _mm256_add_epi32(sum[2 * j], _mm256_madd_epi16(_mm256_mullo_epi16(a, wa), a));
            sum[2 * j + 1] = _mm256_add_epi32(sum[2 * j + 1], _mm256_madd_epi16(_mm256_mullo_epi16(b, wb), b));
        }
    }
    __m256i merged = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(sum[0], sum[1]), _mm256_add_epi32(sum[2], sum[3])),
        _mm256_add_epi32(_mm256_add_epi32(sum[4], sum[5]), _mm256_add_epi32(sum[6], sum[7])));
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), merged);
    for (int lane : lanes) total += lane;
    return static_cast<int>((total * SCALE) /
                            (static_cast<int64_t>(QA) * QA * 64));
}
#endif

void update_row(int16_t* dst, const int16_t* row, bool add) {
#if defined(WALLE_X86_AVX2_DISPATCH)
    if (have_avx2()) {
        update_row_avx2(dst, row, add);
        return;
    }
#endif
#if defined(__AVX512BW__) && defined(__AVX512F__)
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i value = _mm512_load_si512(reinterpret_cast<const __m512i*>(dst + i));
        const __m512i weights = _mm512_load_si512(reinterpret_cast<const __m512i*>(row + i));
        value = add ? _mm512_adds_epi16(value, weights) : _mm512_subs_epi16(value, weights);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + i), value);
    }
#elif defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i weights = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
        value = add ? _mm256_adds_epi16(value, weights) : _mm256_subs_epi16(value, weights);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), value);
    }
#elif defined(WALLE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t value = vld1q_s16(dst + i);
        const int16x8_t weights = vld1q_s16(row + i);
        vst1q_s16(dst + i, add ? vqaddq_s16(value, weights) : vqsubq_s16(value, weights));
    }
#else
    const int sign = add ? 1 : -1;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        dst[i] = static_cast<int16_t>(std::clamp(
            static_cast<int>(dst[i]) + sign * row[i], -32768, 32767));
#endif
}

// The output layer has one weight row per perspective: row 0 weights the
// accumulator of the side to move, row 1 the opponent's. Together they evaluate
// "how good is this for the side to move", and the bundled networks put far more
// into that side-to-move (initiative) term than real chess does: about +1.1 pawns
// in the opening, measured on all three, against the +0.2 a normal evaluation
// carries. Because negamax flips the sign of the score at every ply, that term
// makes the value of a position depend on the parity of the search, so the score a
// GUI draws swings by about a pawn from one depth to the next, and from one
// reviewed move to the next, while the position itself stays the same.
//
// evaluate() therefore takes half of that term back, using the value the network
// reports on a symmetric board as the measurement. infer() stays a plain, faithful
// implementation of the network.
int infer(const Accumulator& acc, int stm) {
#if defined(WALLE_X86_AVX2_DISPATCH)
    if (have_avx2()) return infer_avx2(acc, stm);
#endif
    const int16_t* us = acc.values[stm];
    const int16_t* them = acc.values[stm ^ 1];
    int64_t total = static_cast<int64_t>(network->outputBias) * QA;

#if defined(__AVX512BW__) && defined(__AVX512F__)
    __m512i sum = _mm512_setzero_si512();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i maxValue = _mm512_set1_epi16(QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i a = _mm512_load_si512(reinterpret_cast<const __m512i*>(us + i));
        __m512i b = _mm512_load_si512(reinterpret_cast<const __m512i*>(them + i));
        a = _mm512_min_epi16(_mm512_max_epi16(a, zero), maxValue);
        b = _mm512_min_epi16(_mm512_max_epi16(b, zero), maxValue);
        const __m512i wa = _mm512_load_si512(reinterpret_cast<const __m512i*>(network->outputWeights[0] + i));
        const __m512i wb = _mm512_load_si512(reinterpret_cast<const __m512i*>(network->outputWeights[1] + i));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_mullo_epi16(a, wa), a));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_mullo_epi16(b, wb), b));
    }
    alignas(64) int32_t lanes[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), sum);
    for (int lane : lanes) total += lane;
#elif defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    const __m256i maxValue = _mm256_set1_epi16(QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(us + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(them + i));
        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), maxValue);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), maxValue);
        const __m256i wa = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[0] + i));
        const __m256i wb = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[1] + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(a, wa), a));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(b, wb), b));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
    for (int lane : lanes) total += lane;
#elif defined(WALLE_NEON)
    // Mirrors the x86 path: the products are taken in wrapping int16, then
    // accumulated into 32-bit lanes. Same arithmetic, same rounding.
    // Eight independent accumulators keep several multiply chains in flight
    // and cap each lane at 32 products, far from the int32 limit.
    int32x4_t sum[8];
    for (int s = 0; s < 8; ++s) sum[s] = vdupq_n_s32(0);
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t maxValue = vdupq_n_s16((int16_t)QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        for (int j = 0; j < 4; ++j) {
            const int k = i + 8 * j;
            const int16x8_t a = vminq_s16(vmaxq_s16(vld1q_s16(us + k), zero), maxValue);
            const int16x8_t b = vminq_s16(vmaxq_s16(vld1q_s16(them + k), zero), maxValue);
            const int16x8_t wa = vld1q_s16(network->outputWeights[0] + k);
            const int16x8_t wb = vld1q_s16(network->outputWeights[1] + k);
            const int16x8_t pa = vmulq_s16(a, wa);
            const int16x8_t pb = vmulq_s16(b, wb);
            int32x4_t& sa = sum[2 * j];
            int32x4_t& sb = sum[2 * j + 1];
            sa = vmlal_s16(sa, vget_low_s16(pa), vget_low_s16(a));
            sa = vmlal_s16(sa, vget_high_s16(pa), vget_high_s16(a));
            sb = vmlal_s16(sb, vget_low_s16(pb), vget_low_s16(b));
            sb = vmlal_s16(sb, vget_high_s16(pb), vget_high_s16(b));
        }
    }
    int32x4_t sumAll = vaddq_s32(vaddq_s32(vaddq_s32(sum[0], sum[1]), vaddq_s32(sum[2], sum[3])),
                                 vaddq_s32(vaddq_s32(sum[4], sum[5]), vaddq_s32(sum[6], sum[7])));
#if defined(__aarch64__)
    total += vaddvq_s32(sumAll);
#else
    alignas(16) int32_t lanes[4];
    vst1q_s32(lanes, sumAll);
    for (int lane : lanes) total += lane;
#endif
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        const int32_t a = std::clamp(static_cast<int>(us[i]), 0, QA);
        const int32_t b = std::clamp(static_cast<int>(them[i]), 0, QA);
        total += static_cast<int64_t>(a) * a * network->outputWeights[0][i];
        total += static_cast<int64_t>(b) * b * network->outputWeights[1][i];
    }
#endif
    return static_cast<int>((total * SCALE) /
                            (static_cast<int64_t>(QA) * QA * 64));
}

// Sanity probe on a realistic position. The network drives every evaluation, so a
// corrupt or mismatched file would silently ruin the engine; the probe has to
// reject that without rejecting a good network. Bare-kings material probes are
// not usable: a lone minor piece, or a pawn that cannot promote, is a dead draw,
// so a correctly trained network scores it near zero and the ordering test fails
// on a good file. Take the start position instead and remove one white piece at a
// time: the deltas must look like real material values.
bool network_sane() {
    if (!network) return false;

    const uint8_t backRank[8] = {4, 2, 3, 5, 6, 3, 2, 4};   // R N B Q K B N R
    uint8_t board[64] = {};
    for (int file = 0; file < 8; ++file) {
        board[file] = backRank[file];
        board[8 + file] = 1;   // white pawns
        board[48 + file] = 9;  // black pawns
        board[56 + file] = static_cast<uint8_t>(backRank[file] + 8);
    }

    Accumulator acc{};
    reset(acc, board);
    const int start = infer(acc, 0);
    if (start < -400 || start > 400) return false;

    // a2 pawn, b1 knight, c1 bishop, a1 rook, d1 queen
    const int pieceSquare[5] = {8, 1, 2, 0, 3};
    int val[5];
    for (int t = 0; t < 5; ++t) {
        const int sq = pieceSquare[t];
        const uint8_t saved = board[sq];
        board[sq] = 0;
        reset(acc, board);
        val[t] = start - infer(acc, 0);   // positive: removing the piece costs this much
        board[sq] = saved;
    }

    if (val[0] < 20) return false;                                  // a pawn must be worth something
    if (val[1] <= val[0] || val[2] <= val[0]) return false;         // minors beat a pawn
    if (val[3] <= val[1] || val[3] <= val[2]) return false;         // a rook beats the minors
    if (val[4] <= val[3]) return false;                             // a queen beats a rook

    // The probe board is symmetric, so `start` is exactly the bonus this network
    // gives the side to move -- about a pawn for the bundled ones, several times
    // what real chess puts there. Halving it is the correction evaluate() needs
    // to keep consecutive plies from disagreeing by that bonus.
    network->tempoCorrection = start / 2;

    return val[4] >= 300 && val[4] <= 2500;                         // and on a sane scale
}
} // namespace

bool load(const std::string& networkPath) { return load_network(networkPath, MIDDLEGAME_NET); }

bool load_embedded(int slot) {
#if defined(WALLE_EMBEDDED_NET)
    switch (slot) {
        case MIDDLEGAME_NET:
            return load_network_memory(walle_nnue_start, static_cast<std::size_t>(walle_nnue_end - walle_nnue_start), MIDDLEGAME_NET);
#if defined(WALLE_PHASE_NETS)
        case ENDGAME_NET:
            return load_network_memory(walle_nnue_endgame_start, static_cast<std::size_t>(walle_nnue_endgame_end - walle_nnue_endgame_start), ENDGAME_NET);
        case AGGRESSIVE_NET:
            return load_network_memory(walle_nnue_aggressive_start, static_cast<std::size_t>(walle_nnue_aggressive_end - walle_nnue_aggressive_start), AGGRESSIVE_NET);
#endif
        default:
            return false;
    }
#else
    (void)slot;
    return false;
#endif
}

bool load_embedded() { return load_embedded(MIDDLEGAME_NET); }

bool select(int slot) {
    if (slot < 0 || slot >= NUM_NETS || !owned[slot]) return false;
    network = owned[slot].get();
    return true;
}

bool available() { return network != nullptr; }

// Apply every add and sub of one move to both perspectives in a single pass:
// load the source accumulator once, apply the changed feature rows, store the
// destination once. A quiet move changes two features, a capture three, a
// castle four, so the old per-feature loop streamed the accumulator two to four
// times per move. The add-then-sub order matches Patricia's update kernel (MIT),
// whose networks this engine uses. dst may alias src.
template <int NA, int NS>
inline void update_rows(int16_t* dstA, int16_t* dstB,
                        const int16_t* srcA, const int16_t* srcB,
                        const int16_t* const* addA, const int16_t* const* subA,
                        const int16_t* const* addB, const int16_t* const* subB) {
#if defined(WALLE_X86_AVX2_DISPATCH)
    if (have_avx2()) {
        update_rows_avx2<NA, NS>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
        return;
    }
#endif
#if defined(__AVX512BW__) && defined(__AVX512F__)
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i vA = _mm512_load_si512(reinterpret_cast<const __m512i*>(srcA + i));
        __m512i vB = _mm512_load_si512(reinterpret_cast<const __m512i*>(srcB + i));
        for (int a = 0; a < NA; ++a) {
            vA = _mm512_adds_epi16(vA, _mm512_load_si512(reinterpret_cast<const __m512i*>(addA[a] + i)));
            vB = _mm512_adds_epi16(vB, _mm512_load_si512(reinterpret_cast<const __m512i*>(addB[a] + i)));
        }
        for (int s = 0; s < NS; ++s) {
            vA = _mm512_subs_epi16(vA, _mm512_load_si512(reinterpret_cast<const __m512i*>(subA[s] + i)));
            vB = _mm512_subs_epi16(vB, _mm512_load_si512(reinterpret_cast<const __m512i*>(subB[s] + i)));
        }
        _mm512_store_si512(reinterpret_cast<__m512i*>(dstA + i), vA);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dstB + i), vB);
    }
#elif defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i vA = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcA + i));
        __m256i vB = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcB + i));
        for (int a = 0; a < NA; ++a) {
            vA = _mm256_adds_epi16(vA, _mm256_load_si256(reinterpret_cast<const __m256i*>(addA[a] + i)));
            vB = _mm256_adds_epi16(vB, _mm256_load_si256(reinterpret_cast<const __m256i*>(addB[a] + i)));
        }
        for (int s = 0; s < NS; ++s) {
            vA = _mm256_subs_epi16(vA, _mm256_load_si256(reinterpret_cast<const __m256i*>(subA[s] + i)));
            vB = _mm256_subs_epi16(vB, _mm256_load_si256(reinterpret_cast<const __m256i*>(subB[s] + i)));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstA + i), vA);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstB + i), vB);
    }
#elif defined(WALLE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        int16x8_t vA0 = vld1q_s16(srcA + i),     vA1 = vld1q_s16(srcA + i + 8);
        int16x8_t vB0 = vld1q_s16(srcB + i),     vB1 = vld1q_s16(srcB + i + 8);
        for (int a = 0; a < NA; ++a) {
            vA0 = vqaddq_s16(vA0, vld1q_s16(addA[a] + i));
            vA1 = vqaddq_s16(vA1, vld1q_s16(addA[a] + i + 8));
            vB0 = vqaddq_s16(vB0, vld1q_s16(addB[a] + i));
            vB1 = vqaddq_s16(vB1, vld1q_s16(addB[a] + i + 8));
        }
        for (int s = 0; s < NS; ++s) {
            vA0 = vqsubq_s16(vA0, vld1q_s16(subA[s] + i));
            vA1 = vqsubq_s16(vA1, vld1q_s16(subA[s] + i + 8));
            vB0 = vqsubq_s16(vB0, vld1q_s16(subB[s] + i));
            vB1 = vqsubq_s16(vB1, vld1q_s16(subB[s] + i + 8));
        }
        vst1q_s16(dstA + i, vA0);
        vst1q_s16(dstA + i + 8, vA1);
        vst1q_s16(dstB + i, vB0);
        vst1q_s16(dstB + i + 8, vB1);
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int vA = srcA[i], vB = srcB[i];
        for (int a = 0; a < NA; ++a) {
            vA = std::clamp(vA + addA[a][i], -32768, 32767);
            vB = std::clamp(vB + addB[a][i], -32768, 32767);
        }
        for (int s = 0; s < NS; ++s) {
            vA = std::clamp(vA - subA[s][i], -32768, 32767);
            vB = std::clamp(vB - subB[s][i], -32768, 32767);
        }
        dstA[i] = static_cast<int16_t>(vA);
        dstB[i] = static_cast<int16_t>(vB);
    }
#endif
}

inline const int16_t* update_row_for(const PieceChange& piece, int perspective) {
    const int pc = (piece.color << 3) | piece.type;
    return network->featureWeights[feature_index(pc, piece.square, perspective)];
}

void apply_update(Accumulator& dst, const Accumulator& src, const Update& update) {
    if (!network) return;
    if (update.adds == 0 && update.subs == 0) {
        if (&dst != &src) dst = src;
        return;
    }

    const int16_t* addA[2] = {nullptr, nullptr};
    const int16_t* subA[2] = {nullptr, nullptr};
    const int16_t* addB[2] = {nullptr, nullptr};
    const int16_t* subB[2] = {nullptr, nullptr};
    for (int i = 0; i < update.adds; ++i) {
        addA[i] = update_row_for(update.add[i], 0);
        addB[i] = update_row_for(update.add[i], 1);
    }
    for (int i = 0; i < update.subs; ++i) {
        subA[i] = update_row_for(update.sub[i], 0);
        subB[i] = update_row_for(update.sub[i], 1);
    }

    const int16_t* srcA = src.values[0];
    const int16_t* srcB = src.values[1];
    int16_t* dstA = dst.values[0];
    int16_t* dstB = dst.values[1];
    if (update.adds == 0) {
        if (update.subs == 1) update_rows<0, 1>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
        else                  update_rows<0, 2>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
    } else if (update.adds == 1) {
        if (update.subs == 1) update_rows<1, 1>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
        else                  update_rows<1, 2>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
    } else {
        if (update.subs == 1) update_rows<2, 1>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
        else                  update_rows<2, 2>(dstA, dstB, srcA, srcB, addA, subA, addB, subB);
    }
}

void revert_update(Accumulator& dst, const Accumulator& src, const Update& update) {
    if (!network) return;
    Update inverse;
    for (int i = 0; i < update.adds; ++i) inverse.subPiece(update.add[i].color, update.add[i].type, update.add[i].square);
    for (int i = 0; i < update.subs; ++i) inverse.addPiece(update.sub[i].color, update.sub[i].type, update.sub[i].square);
    apply_update(dst, src, inverse);
}

void prefetch_update(const Update& update) {
    if (!network) return;
#if defined(__GNUC__) || defined(__clang__)
    for (int i = 0; i < update.adds; ++i) {
        __builtin_prefetch(update_row_for(update.add[i], 0), 0, 3);
        __builtin_prefetch(update_row_for(update.add[i], 1), 0, 3);
    }
    for (int i = 0; i < update.subs; ++i) {
        __builtin_prefetch(update_row_for(update.sub[i], 0), 0, 3);
        __builtin_prefetch(update_row_for(update.sub[i], 1), 0, 3);
    }
#else
    (void)update;
#endif
}

void reset(Accumulator& acc, const uint8_t board[64]) {
    if (!network) return;
    for (int perspective = 0; perspective < 2; ++perspective) {
        int16_t* dst = acc.values[perspective];
        std::copy_n(network->bias, HIDDEN_SIZE, dst);
        for (int square = 0; square < 64; ++square) {
            const int piece = board[square];
            if (!piece) continue;
            const int index = feature_index(piece, square, perspective);
            update_row(dst, network->featureWeights[index], true);
        }
    }
}

bool consistent(const Accumulator& accumulator, const uint8_t board[64]) {
    if (!network) return true;
    Accumulator expected{};
    reset(expected, board);
    return std::memcmp(&expected, &accumulator, sizeof(expected)) == 0;
}

int evaluate(const Accumulator& accumulator, int sideToMove) {
    if (!network) return 0;
    return infer(accumulator, sideToMove) - network->tempoCorrection;
}
} // namespace Nnue
