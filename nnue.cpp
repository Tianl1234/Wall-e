#include "nnue.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
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

void update_row(int16_t* dst, const int16_t* row, bool add) {
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

int infer(const Accumulator& acc, int stm) {
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
    // Two accumulators halve the per-lane product count, keeping the 32-bit
    // lanes overflow-safe (256 products x 255 x 255 x 128 < 2^31), the same
    // bound the AVX2 path naturally has.
    int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t maxValue = vdupq_n_s16((int16_t)QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        for (int j = 0; j < 2; ++j) {
            const int k = i + 8 * j;
            const int16x8_t a = vminq_s16(vmaxq_s16(vld1q_s16(us + k), zero), maxValue);
            const int16x8_t b = vminq_s16(vmaxq_s16(vld1q_s16(them + k), zero), maxValue);
            const int16x8_t wa = vld1q_s16(network->outputWeights[0] + k);
            const int16x8_t wb = vld1q_s16(network->outputWeights[1] + k);
            const int16x8_t pa = vmulq_s16(a, wa);
            const int16x8_t pb = vmulq_s16(b, wb);
            int32x4_t& s = j ? sum1 : sum0;
            s = vmlal_s16(s, vget_low_s16(pa), vget_low_s16(a));
            s = vmlal_s16(s, vget_high_s16(pa), vget_high_s16(a));
            s = vmlal_s16(s, vget_low_s16(pb), vget_low_s16(b));
            s = vmlal_s16(s, vget_high_s16(pb), vget_high_s16(b));
        }
    }
    const int32x4_t sum = vaddq_s32(sum0, sum1);
#if defined(__aarch64__)
    total += vaddvq_s32(sum);
#else
    alignas(16) int32_t lanes[4];
    vst1q_s32(lanes, sum);
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

// Apply one change to both perspectives in a single pass. The two rows and the
// two accumulator streams are independent, so interleaving them keeps twice as
// much work in flight as two separate passes while still writing each
// accumulator exactly once. (Measured: bench 12 goes from ~327 ms to ~299 ms
// with identical node counts.)
void update_row2(int16_t* dstA, int16_t* dstB, const int16_t* rowA, const int16_t* rowB, bool add) {
#if defined(__AVX512BW__) && defined(__AVX512F__)
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        const __m512i wA = _mm512_load_si512(reinterpret_cast<const __m512i*>(rowA + i));
        const __m512i wB = _mm512_load_si512(reinterpret_cast<const __m512i*>(rowB + i));
        const __m512i vA = _mm512_load_si512(reinterpret_cast<const __m512i*>(dstA + i));
        const __m512i vB = _mm512_load_si512(reinterpret_cast<const __m512i*>(dstB + i));
        _mm512_store_si512(reinterpret_cast<__m512i*>(dstA + i),
                           add ? _mm512_adds_epi16(vA, wA) : _mm512_subs_epi16(vA, wA));
        _mm512_store_si512(reinterpret_cast<__m512i*>(dstB + i),
                           add ? _mm512_adds_epi16(vB, wB) : _mm512_subs_epi16(vB, wB));
    }
#elif defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        const __m256i wA = _mm256_load_si256(reinterpret_cast<const __m256i*>(rowA + i));
        const __m256i wB = _mm256_load_si256(reinterpret_cast<const __m256i*>(rowB + i));
        const __m256i vA = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstA + i));
        const __m256i vB = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstB + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstA + i),
                           add ? _mm256_adds_epi16(vA, wA) : _mm256_subs_epi16(vA, wA));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstB + i),
                           add ? _mm256_adds_epi16(vB, wB) : _mm256_subs_epi16(vB, wB));
    }
#elif defined(WALLE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        const int16x8_t wA0 = vld1q_s16(rowA + i),      wA1 = vld1q_s16(rowA + i + 8);
        const int16x8_t wB0 = vld1q_s16(rowB + i),      wB1 = vld1q_s16(rowB + i + 8);
        const int16x8_t vA0 = vld1q_s16(dstA + i),      vA1 = vld1q_s16(dstA + i + 8);
        const int16x8_t vB0 = vld1q_s16(dstB + i),      vB1 = vld1q_s16(dstB + i + 8);
        vst1q_s16(dstA + i,     add ? vqaddq_s16(vA0, wA0) : vqsubq_s16(vA0, wA0));
        vst1q_s16(dstA + i + 8, add ? vqaddq_s16(vA1, wA1) : vqsubq_s16(vA1, wA1));
        vst1q_s16(dstB + i,     add ? vqaddq_s16(vB0, wB0) : vqsubq_s16(vB0, wB0));
        vst1q_s16(dstB + i + 8, add ? vqaddq_s16(vB1, wB1) : vqsubq_s16(vB1, wB1));
    }
#else
    const int sign = add ? 1 : -1;
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        dstA[i] = static_cast<int16_t>(std::clamp(static_cast<int>(dstA[i]) + sign * rowA[i], -32768, 32767));
        dstB[i] = static_cast<int16_t>(std::clamp(static_cast<int>(dstB[i]) + sign * rowB[i], -32768, 32767));
    }
#endif
}

void apply_changes(Accumulator& acc, const PieceChange* changes, int count) {
    if (!network || count <= 0) return;
    for (int change = 0; change < count; ++change) {
        const PieceChange& piece = changes[change];
        if (piece.type < 1 || piece.type > 6 || piece.square < 0 || piece.square >= 64) continue;
        const int pc = (piece.color << 3) | piece.type;
        update_row2(acc.values[0], acc.values[1],
                    network->featureWeights[feature_index(pc, piece.square, 0)],
                    network->featureWeights[feature_index(pc, piece.square, 1)],
                    piece.add);
    }
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
    return network ? infer(accumulator, sideToMove) : 0;
}
} // namespace Nnue
