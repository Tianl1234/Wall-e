#include "nnue.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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
std::unique_ptr<Network> network;

// Only the Android package defines WALLE_EMBEDDED_NET. The network is appended to
// the engine binary there (see nnue_embedded.cpp) because a GUI executes the engine
// out of the APK's lib directory, where there is no working directory to hold
// nnue/fingolfin.nnue.
#if defined(WALLE_EMBEDDED_NET)
extern "C" const unsigned char walle_nnue_start[];
extern "C" const unsigned char walle_nnue_end[];
#endif

bool load_network_memory(const void* data, std::size_t size) {
    if (data == nullptr || size < NETWORK_BYTES) return false;
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
    network = std::move(candidate);
    return true;
}

bool load_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::vector<int16_t> raw(NETWORK_BYTES / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(raw.data()), NETWORK_BYTES);
    if (file.gcount() != static_cast<std::streamsize>(NETWORK_BYTES)) return false;
    return load_network_memory(raw.data(), NETWORK_BYTES);
}

inline int feature_index(int piece, int square, int perspective) {
    const int color = piece >> 3;
    const int type = piece & 7;
    const int orientedSquare = square ^ (perspective ? 56 : 0);
    return ((color ^ perspective) * 6 + type - 1) * 64 + orientedSquare;
}

void add_row(int16_t* dst, const int16_t* row) {
#if defined(__AVX512BW__) && defined(__AVX512F__)
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i value = _mm512_load_si512(reinterpret_cast<const __m512i*>(dst + i));
        const __m512i weights = _mm512_load_si512(reinterpret_cast<const __m512i*>(row + i));
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_adds_epi16(value, weights));
    }
#elif defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i weights = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_adds_epi16(value, weights));
    }
#elif defined(WALLE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t value = vld1q_s16(dst + i);
        const int16x8_t weights = vld1q_s16(row + i);
        vst1q_s16(dst + i, vqaddq_s16(value, weights));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        dst[i] = static_cast<int16_t>(std::clamp(
            static_cast<int>(dst[i]) + row[i], -32768, 32767));
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
    int32x4_t sum = vdupq_n_s32(0);
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t maxValue = vdupq_n_s16((int16_t)QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t a = vminq_s16(vmaxq_s16(vld1q_s16(us + i), zero), maxValue);
        const int16x8_t b = vminq_s16(vmaxq_s16(vld1q_s16(them + i), zero), maxValue);
        const int16x8_t wa = vld1q_s16(network->outputWeights[0] + i);
        const int16x8_t wb = vld1q_s16(network->outputWeights[1] + i);
        const int16x8_t pa = vmulq_s16(a, wa);
        const int16x8_t pb = vmulq_s16(b, wb);
        sum = vmlal_s16(sum, vget_low_s16(pa), vget_low_s16(a));
        sum = vmlal_s16(sum, vget_high_s16(pa), vget_high_s16(a));
        sum = vmlal_s16(sum, vget_low_s16(pb), vget_low_s16(b));
        sum = vmlal_s16(sum, vget_high_s16(pb), vget_high_s16(b));
    }
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
} // namespace

bool load(const std::string& networkPath) { return load_network(networkPath); }

bool load_embedded() {
#if defined(WALLE_EMBEDDED_NET)
    const std::size_t size = static_cast<std::size_t>(walle_nnue_end - walle_nnue_start);
    return load_network_memory(walle_nnue_start, size);
#else
    return false;
#endif
}

bool available() { return static_cast<bool>(network); }

void apply_changes(Accumulator& acc, const PieceChange* changes, int count) {
    if (!network || count <= 0) return;
    alignas(64) int32_t delta[2][HIDDEN_SIZE]{};
    for (int change = 0; change < count; ++change) {
        const PieceChange& piece = changes[change];
        if (piece.type < 1 || piece.type > 6 || piece.square < 0 || piece.square >= 64) continue;
        const int pc = (piece.color << 3) | piece.type;
        const int sign = piece.add ? 1 : -1;
        for (int perspective = 0; perspective < 2; ++perspective) {
            const int index = feature_index(pc, piece.square, perspective);
            const int16_t* weights = network->featureWeights[index];
            int32_t* changesForSide = delta[perspective];
#if defined(__AVX512BW__) && defined(__AVX512F__)
            for (int i = 0; i < HIDDEN_SIZE; i += 16) {
                const __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
                const __m512i expanded = _mm512_cvtepi16_epi32(packed);
                const __m512i signedWeights = _mm512_mullo_epi32(expanded, _mm512_set1_epi32(sign));
                __m512i old = _mm512_load_si512(reinterpret_cast<const __m512i*>(changesForSide + i));
                _mm512_store_si512(reinterpret_cast<__m512i*>(changesForSide + i), _mm512_add_epi32(old, signedWeights));
            }
#elif defined(WALLE_NEON)
            for (int i = 0; i < HIDDEN_SIZE; i += 8) {
                const int16x8_t packed = vld1q_s16(weights + i);
                int32x4_t lo = vmovl_s16(vget_low_s16(packed));
                int32x4_t hi = vmovl_s16(vget_high_s16(packed));
                if (sign < 0) { lo = vnegq_s32(lo); hi = vnegq_s32(hi); }
                vst1q_s32(changesForSide + i, vaddq_s32(vld1q_s32(changesForSide + i), lo));
                vst1q_s32(changesForSide + i + 4, vaddq_s32(vld1q_s32(changesForSide + i + 4), hi));
            }
#else
            for (int i = 0; i < HIDDEN_SIZE; ++i) changesForSide[i] += sign * weights[i];
#endif
        }
    }

    for (int perspective = 0; perspective < 2; ++perspective) {
        int16_t* dst = acc.values[perspective];
#if defined(__AVX512BW__) && defined(__AVX512F__)
        const __m512i minValue = _mm512_set1_epi32(-32768);
        const __m512i maxValue = _mm512_set1_epi32(32767);
        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            const __m256i old16 = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
            const __m512i old32 = _mm512_cvtepi16_epi32(old16);
            const __m512i change = _mm512_load_si512(reinterpret_cast<const __m512i*>(delta[perspective] + i));
            const __m512i result = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(old32, change), minValue), maxValue);
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm512_cvtsepi32_epi16(result));
        }
#elif defined(WALLE_NEON)
        for (int i = 0; i < HIDDEN_SIZE; i += 8) {
            const int16x8_t old16 = vld1q_s16(dst + i);
            const int32x4_t low = vaddq_s32(vmovl_s16(vget_low_s16(old16)), vld1q_s32(delta[perspective] + i));
            const int32x4_t high = vaddq_s32(vmovl_s16(vget_high_s16(old16)), vld1q_s32(delta[perspective] + i + 4));
            vst1q_s16(dst + i, vcombine_s16(vqmovn_s32(low), vqmovn_s32(high)));   // saturating narrow == clamp
        }
#else
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            dst[i] = static_cast<int16_t>(std::clamp(
                static_cast<int32_t>(dst[i]) + delta[perspective][i], -32768, 32767));
#endif
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
            add_row(dst, network->featureWeights[index]);
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
