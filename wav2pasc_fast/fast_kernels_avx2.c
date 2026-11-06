/*
 * fast_kernels_avx2.c — AVX2 kernels for wav2pasc_fast (x86 only).
 *
 * This file is compiled with the AVX2 instruction set enabled
 * (-mavx2 on GCC/Clang, /arch:AVX2 on MSVC) and MUST be compiled with
 * floating-point contraction disabled (-ffp-contract=off; MSVC never
 * contracts under the default /fp:precise) so that every
 * "vmulpd; vaddpd" pair stays two separately rounded operations, exactly
 * as the SSE2 reference arithmetic does.  FMA must never be emitted:
 * it would change rounding and break byte-exactness.
 *
 * Bit-exactness strategy:
 *   - vector lanes carry independent outputs (4 different k in the
 *     convolver, 4 different subbands m / i elsewhere);
 *   - the reduction order inside each lane is the original one
 *     (j = 0..7 for the coefficient convolver, k = 0..63 for the cosine
 *     transform, j = 0..11 for energies, j = 0..31 for the threshold
 *     sums), accumulated in the same accumulator the scalar code uses,
 *     started from the same +0.0;
 *   - vmulpd/vaddpd/vdivpd/vmaxpd/vroundpd (RN)/vcvtpd2dq of integral
 *     values and the (<<8)^0x80000000 >>16 integer twiddle are bit-wise
 *     equivalent to the scalar SSE2/x87 sequence.
 */

#include "fast_kernels.h"

#ifdef FK_X86

#include <immintrin.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER)
#define FK_CMP_LT _MM_CMPINT_LT
#else
#define FK_CMP_LT _CMP_LT_OS
#endif

#define ZCLAMP 0.9999998807907104
#define SC16   3.0517578125e-05        /* 2^-15, exact */
#define SC23   8388608.0               /* 2^23         */

/* ------------------------------------------------------------------ */

static void fk_analysis(const double *pcmL, const double *pcmR,
                        double *xL, double *xR,
                        const double *coeff, const double *cosT,
                        double *SL, double *SR)
{
    double ZL[64], ZR[64];
    int b, k, g, j;

    memcpy(xL + FK_XHIST, pcmL, FK_NBLK * 32 * sizeof(double));
    memcpy(xR + FK_XHIST, pcmR, FK_NBLK * 32 * sizeof(double));

    for (b = 0; b < FK_NBLK; b++) {
        const int base = FK_XHIST + 32 * b + 31;

        /* Z[k] = sum_{j=0..7} xbuf[base - 64j - k] * coeff[64j + k].
           4 consecutive k per lane; xbuf loaded reversed so lane r holds
           k = kg*4 + r (matches the ascending-k scalar order). */
        for (g = 0; g < 16; g++) {
            __m256d aL = _mm256_setzero_pd();
            __m256d aR = _mm256_setzero_pd();
            const int kq = g * 4;
            for (j = 0; j < 8; j++) {
                const int idx = 64 * j + kq;
                __m256d xl = _mm256_permute4x64_pd(
                    _mm256_loadu_pd(xL + base - idx - 3), 0x1B);
                __m256d xr = _mm256_permute4x64_pd(
                    _mm256_loadu_pd(xR + base - idx - 3), 0x1B);
                __m256d c = _mm256_loadu_pd(coeff + idx);
                aL = _mm256_add_pd(aL, _mm256_mul_pd(xl, c));
                aR = _mm256_add_pd(aR, _mm256_mul_pd(xr, c));
            }
            _mm256_storeu_pd(ZL + kq, aL);
            _mm256_storeu_pd(ZR + kq, aR);
        }

        /* S[m] = sum_{k=0..63} cosT[k*32 + m] * Z[k]; 4 m per lane. */
        for (g = 0; g < 8; g++) {
            __m256d sL = _mm256_setzero_pd();
            __m256d sR = _mm256_setzero_pd();
            const int mq = g * 4;
            for (k = 0; k < 64; k++) {
                __m256d c = _mm256_loadu_pd(cosT + k * 32 + mq);
                sL = _mm256_add_pd(sL,
                        _mm256_mul_pd(c, _mm256_set1_pd(ZL[k])));
                sR = _mm256_add_pd(sR,
                        _mm256_mul_pd(c, _mm256_set1_pd(ZR[k])));
            }
            _mm256_storeu_pd(SL + b * 32 + mq, sL);
            _mm256_storeu_pd(SR + b * 32 + mq, sR);
        }
    }

    memmove(xL, xL + FK_NBLK * 32, FK_XHIST * sizeof(double));
    memmove(xR, xR + FK_NBLK * 32, FK_XHIST * sizeof(double));
}

static void fk_energy_max(const double *S, double *e, double *mx)
{
    const __m256d amask = _mm256_castsi256_pd(
        _mm256_set1_epi64x((long long)0x7FFFFFFFFFFFFFFF));
    for (int g = 0; g < 8; g++) {
        const int iq = g * 4;
        __m256d acc = _mm256_setzero_pd();
        __m256d m = _mm256_setzero_pd();
        for (int j = 0; j < FK_NBLK; j++) {
            __m256d s = _mm256_loadu_pd(S + j * 32 + iq);
            m = _mm256_max_pd(m, _mm256_and_pd(s, amask));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(s, s));
        }
        _mm256_storeu_pd(e + iq, acc);
        _mm256_storeu_pd(mx + iq, m);
    }
}

static void fk_threshold(const double *e, const double *spreadT,
                         double K1, const double *absM, double *thr)
{
    const __m256d k1v = _mm256_set1_pd(K1);
    for (int g = 0; g < 8; g++) {
        const int iq = g * 4;
        __m256d t = _mm256_mul_pd(k1v, _mm256_loadu_pd(absM + iq));
        for (int j = 0; j < 32; j++)
            t = _mm256_add_pd(t,
                    _mm256_mul_pd(_mm256_loadu_pd(spreadT + j * 32 + iq),
                                  _mm256_set1_pd(e[j])));
        _mm256_storeu_pd(thr + iq, t);
    }
}

static void fk_band_max(const double *S, double *mx)
{
    const __m256d amask = _mm256_castsi256_pd(
        _mm256_set1_epi64x((long long)0x7FFFFFFFFFFFFFFF));
    for (int g = 0; g < 8; g++) {
        const int iq = g * 4;
        __m256d m = _mm256_setzero_pd();
        for (int j = 0; j < FK_NBLK; j++) {
            __m256d s = _mm256_loadu_pd(S + j * 32 + iq);
            m = _mm256_max_pd(m, _mm256_and_pd(s, amask));
        }
        _mm256_storeu_pd(mx + iq, m);
    }
}

/*
 * n == 32 fast path (all lanes):
 *   q = round2even( 2^23 * (omv*(qs clamped) - off) ), qs = S/scl
 * the scalar order per subband is preserved lane-wise.
 */
static void fk_quantize_row(const double *S, int32_t n,
                            const double *scl, const double *omv, const double *off,
                            const int32_t *amask, int32_t *code)
{
    if (n == 32) {
        /* Compare against 1.0 (not ZCLAMP) — matches the scalar
         * `if (!(qs < 1.0)) qs = ZCLAMP;`.  The previous code reused
         * the ZCLAMP constant for the comparison too, which clamped
         * values in [ZCLAMP, 1.0) that the scalar kernel keeps. */
        const __m256d threshold = _mm256_set1_pd(1.0);
        const __m256d clamp = _mm256_set1_pd(ZCLAMP);
        const __m256d c23 = _mm256_set1_pd(SC23);
        const __m128i xh = _mm_set1_epi32((int)0x80000000u);
        for (int g = 0; g < 8; g++) {
            const int iq = g * 4;
            __m256d q = _mm256_div_pd(_mm256_loadu_pd(S + iq),
                                      _mm256_loadu_pd(scl + iq));
            __m256d lt = _mm256_cmp_pd(q, threshold, FK_CMP_LT);
            q = _mm256_blendv_pd(clamp, q, lt);
            q = _mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(omv + iq), q),
                              _mm256_loadu_pd(off + iq));
            q = _mm256_mul_pd(q, c23);
            q = _mm256_round_pd(q, _MM_FROUND_TO_NEAREST_INT |
                                     _MM_FROUND_NO_EXC);
            __m128i v = _mm256_cvtpd_epi32(q);
            v = _mm_slli_epi32(v, 8);
            v = _mm_xor_si128(v, xh);
            v = _mm_srli_epi32(v, 16);
            v = _mm_and_si128(v, _mm_loadu_si128((const __m128i *)(amask + iq)));
            _mm_storeu_si128((__m128i *)(code + iq), v);
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (amask[i]) {
                double qs = S[i] / scl[i];
                if (!(qs < 1.0)) qs = ZCLAMP;
                double q = omv[i] * qs - off[i];
                q = nearbyint(q * SC23);
                long v = (long)q;
                uint32_t u = ((uint32_t)(v & 0xFFFFFFFF) << 8) ^ 0x80000000u;
                code[i] = (int32_t)(u >> 16);
            } else code[i] = 0;
        }
    }
}

static void fk_deint16(const int16_t *ilv, double *L, double *R, uint32_t n)
{
    const __m256d sc = _mm256_set1_pd(SC16);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(ilv + 2 * i));
        __m256i lo = _mm256_srai_epi32(v, 16);                 /* right s16 */
        __m256i hi = _mm256_srai_epi32(_mm256_slli_epi32(v, 16), 16);
        _mm256_storeu_pd(L + i, _mm256_mul_pd(
            sc, _mm256_cvtepi32_pd(_mm256_castsi256_si128(hi))));
        _mm256_storeu_pd(L + i + 4, _mm256_mul_pd(
            sc, _mm256_cvtepi32_pd(_mm256_extracti128_si256(hi, 1))));
        _mm256_storeu_pd(R + i, _mm256_mul_pd(
            sc, _mm256_cvtepi32_pd(_mm256_castsi256_si128(lo))));
        _mm256_storeu_pd(R + i + 4, _mm256_mul_pd(
            sc, _mm256_cvtepi32_pd(_mm256_extracti128_si256(lo, 1))));
    }
    for (; i < n; i++) {
        L[i] = (double)ilv[2 * i + 0] / 32768.0;
        R[i] = (double)ilv[2 * i + 1] / 32768.0;
    }
}

static void fk_deint24(const uint8_t *raw, double *L, double *R, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *p = raw + 6 * i;
        int32_t l = ((int32_t)p[0] << 8 | (int32_t)p[1] << 16 | (int32_t)p[2] << 24) >> 8;
        int32_t r = ((int32_t)p[3] << 8 | (int32_t)p[4] << 16 | (int32_t)p[5] << 24) >> 8;
        L[i] = (double)l / 8388608.0;
        R[i] = (double)r / 8388608.0;
    }
}

const kf_ops kf_ops_avx2 = {
    fk_analysis, fk_energy_max, fk_threshold, fk_band_max,
    fk_quantize_row, fk_deint16, fk_deint24
};

#endif /* FK_X86 */
