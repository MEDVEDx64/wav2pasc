/*
 * fast_kernels_avx512.c — AVX-512F kernels for wav2pasc (x86 only).
 *
 * This file is compiled with the AVX-512F instruction set enabled
 * (-mavx512f on GCC/Clang, /arch:AVX512 on MSVC) and MUST be compiled
 * with floating-point contraction disabled (-ffp-contract=off; MSVC
 * never contracts under the default /fp:precise) so that every
 * "vmulpd; vaddpd" pair stays two separately rounded operations,
 * exactly as the scalar and AVX2 reference arithmetic do.  FMA must
 * never be emitted: it would change rounding and break byte-exactness.
 *
 * Only AVX-512F is required at runtime; the 256-bit integer ops on
 * the cvtpd2dq results use the VEX-encoded AVX2 forms, available
 * whenever AVX-512F is.  No AVX-512VL/DQ/BW-specific intrinsics are
 * used, so a runtime check for AVX-512F alone is sufficient.
 *
 * Bit-exactness strategy (identical to fast_kernels_avx2.c, widened
 * from 4 to 8 lanes per vector):
 *   - vector lanes carry independent outputs (8 different k in the
 *     convolver, 8 different subbands m / i elsewhere);
 *   - the reduction order inside each lane is the original one
 *     (j = 0..7 for the coefficient convolver, k = 0..63 for the
 *     cosine transform, j = 0..11 for energies, j = 0..31 for the
 *     threshold sums), accumulated in the same accumulator the
 *     scalar code uses, started from the same +0.0;
 *   - vmulpd/vaddpd/vdivpd/vmaxpd/vroundpd (RN)/vcvtpd2dq of integral
 *     values and the (<<8)^0x80000000 >>16 integer twiddle are
 *     bit-wise equivalent to the scalar SSE2/x87 sequence.
 */

#include "fast_kernels.h"

#ifdef FK_X86

#include <immintrin.h>
#include <string.h>
#include <math.h>

/*
 * Floating-point compare predicate for less-than, ordered, signalling
 * (imm8 = 0x11).  Used directly as a numeric imm8 so MSVC and GCC
 * emit identical instructions (the AVX2 TU's _MM_CMPINT_LT happens
 * to map to a different predicate value but produces the same mask
 * for non-NaN inputs; here we use the canonical 0x11 on both
 * compilers).
 */
#define FK_CMP_LT 0x11

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
    /* Lane-reversal index for the 8-wide xbuf load (see below). */
    const __m512i rev = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);

    memcpy(xL + FK_XHIST, pcmL, FK_NBLK * 32 * sizeof(double));
    memcpy(xR + FK_XHIST, pcmR, FK_NBLK * 32 * sizeof(double));

    for (int b = 0; b < FK_NBLK; b++) {
        const int base = FK_XHIST + 32 * b + 31;

        /* Z[k] = sum_{j=0..7} xbuf[base - 64j - k] * coeff[64j + k].
          8 consecutive k per lane; xbuf loaded reversed so lane r
          holds k = kg*8 + r (matches the ascending-k scalar order). */
        for (int g = 0; g < 8; g++) {
            const int kq = g * 8;
            __m512d aL = _mm512_setzero_pd();
            __m512d aR = _mm512_setzero_pd();
            for (int j = 0; j < 8; j++) {
                const int idx = 64 * j + kq;
                /* Load x[base - idx - 7 .. base - idx - 0] and reverse
                   so lane r (k = kq + r) gets x[base - idx - r]. */
                __m512d xl = _mm512_permutexvar_pd(rev,
                        _mm512_loadu_pd(xL + base - idx - 7));
                __m512d xr = _mm512_permutexvar_pd(rev,
                        _mm512_loadu_pd(xR + base - idx - 7));
                __m512d c = _mm512_loadu_pd(coeff + idx);
                aL = _mm512_add_pd(aL, _mm512_mul_pd(xl, c));
                aR = _mm512_add_pd(aR, _mm512_mul_pd(xr, c));
            }
            _mm512_storeu_pd(ZL + kq, aL);
            _mm512_storeu_pd(ZR + kq, aR);
        }

        /* S[m] = sum_{k=0..63} cosT[k*32 + m] * Z[k]; 8 m per lane. */
        for (int g = 0; g < 4; g++) {
            const int mq = g * 8;
            __m512d sL = _mm512_setzero_pd();
            __m512d sR = _mm512_setzero_pd();
            for (int k = 0; k < 64; k++) {
                __m512d c = _mm512_loadu_pd(cosT + k * 32 + mq);
                sL = _mm512_add_pd(sL,
                        _mm512_mul_pd(c, _mm512_set1_pd(ZL[k])));
                sR = _mm512_add_pd(sR,
                        _mm512_mul_pd(c, _mm512_set1_pd(ZR[k])));
            }
            _mm512_storeu_pd(SL + b * 32 + mq, sL);
            _mm512_storeu_pd(SR + b * 32 + mq, sR);
        }
    }

    memmove(xL, xL + FK_NBLK * 32, FK_XHIST * sizeof(double));
    memmove(xR, xR + FK_NBLK * 32, FK_XHIST * sizeof(double));
}

static void fk_energy_max(const double *S, double *e, double *mx)
{
    /* abs() via integer AND with 0x7FFF...F (AVX-512F has no
       double-precision abs intrinsic; vpternlogd would also work). */
    const __m512i amask_i = _mm512_set1_epi64((long long)0x7FFFFFFFFFFFFFFF);
    for (int g = 0; g < 4; g++) {
        const int iq = g * 8;
        __m512d acc = _mm512_setzero_pd();
        __m512d m = _mm512_setzero_pd();
        for (int j = 0; j < FK_NBLK; j++) {
            __m512d s = _mm512_loadu_pd(S + j * 32 + iq);
            __m512d absv = _mm512_castsi512_pd(
                _mm512_and_si512(_mm512_castpd_si512(s), amask_i));
            m = _mm512_max_pd(m, absv);
            acc = _mm512_add_pd(acc, _mm512_mul_pd(s, s));
        }
        _mm512_storeu_pd(e + iq, acc);
        _mm512_storeu_pd(mx + iq, m);
    }
}

static void fk_threshold(const double *e, const double *spreadT,
                         double K1, const double *absM, double *thr)
{
    const __m512d k1v = _mm512_set1_pd(K1);
    for (int g = 0; g < 4; g++) {
        const int iq = g * 8;
        __m512d t = _mm512_mul_pd(k1v, _mm512_loadu_pd(absM + iq));
        for (int j = 0; j < 32; j++)
            t = _mm512_add_pd(t,
                    _mm512_mul_pd(_mm512_loadu_pd(spreadT + j * 32 + iq),
                                  _mm512_set1_pd(e[j])));
        _mm512_storeu_pd(thr + iq, t);
    }
}

static void fk_band_max(const double *S, double *mx)
{
    const __m512i amask_i = _mm512_set1_epi64((long long)0x7FFFFFFFFFFFFFFF);
    for (int g = 0; g < 4; g++) {
        const int iq = g * 8;
        __m512d m = _mm512_setzero_pd();
        for (int j = 0; j < FK_NBLK; j++) {
            __m512d s = _mm512_loadu_pd(S + j * 32 + iq);
            __m512d absv = _mm512_castsi512_pd(
                _mm512_and_si512(_mm512_castpd_si512(s), amask_i));
            m = _mm512_max_pd(m, absv);
        }
        _mm512_storeu_pd(mx + iq, m);
    }
}

/*
 * n == 32 fast path (all lanes):
 *   q = round2even( 2^23 * (omv*(qs clamped) - off) ), qs = S/scl
 * the scalar order per subband is preserved lane-wise.
 *
 * The rounding and the double->int32 conversion are fused into a
 * single vcvtpd2dq {rn-sae} instruction; this is bit-exact with the
 * AVX2 path's "vroundpd(rn) then vcvtpd2dq" sequence because both
 * round to nearest-even (the AVX2 path's vroundpd rounds to an
 * integer-valued double, after which vcvtpd2dq is an exact format
 * conversion; the fused AVX-512 form applies the same RN rule
 * directly during conversion).
 */
static void fk_quantize_row(const double *S, int32_t n,
                            const double *scl, const double *omv, const double *off,
                            const int32_t *amask, int32_t *code)
{
    if (n == 32) {
        /* Compare against 1.0 (not ZCLAMP) — matches the scalar
         * `if (!(qs < 1.0)) qs = ZCLAMP;`.  Reusing the ZCLAMP constant
         * for the comparison too (as the AVX2 kernel did before the
         * fix) clamps values in [ZCLAMP, 1.0) that the scalar keeps. */
        const __m512d threshold = _mm512_set1_pd(1.0);
        const __m512d clamp = _mm512_set1_pd(ZCLAMP);
        const __m512d c23 = _mm512_set1_pd(SC23);
        const __m256i xh = _mm256_set1_epi32((int)0x80000000u);
        for (int g = 0; g < 4; g++) {
            const int iq = g * 8;
            __m512d q = _mm512_div_pd(_mm512_loadu_pd(S + iq),
                                      _mm512_loadu_pd(scl + iq));
            __mmask8 lt = _mm512_cmp_pd_mask(q, threshold, FK_CMP_LT);
            q = _mm512_mask_blend_pd(lt, clamp, q);
            q = _mm512_sub_pd(_mm512_mul_pd(_mm512_loadu_pd(omv + iq), q),
                              _mm512_loadu_pd(off + iq));
            q = _mm512_mul_pd(q, c23);
            __m256i v = _mm512_cvt_roundpd_epi32(q,
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm256_slli_epi32(v, 8);
            v = _mm256_xor_si256(v, xh);
            v = _mm256_srli_epi32(v, 16);
            v = _mm256_and_si256(v,
                    _mm256_loadu_si256((const __m256i *)(amask + iq)));
            _mm256_storeu_si256((__m256i *)(code + iq), v);
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
    const __m512d sc = _mm512_set1_pd(SC16);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        /* 16 stereo pairs = 32 int16 = 64 bytes = 1 zmm. */
        __m512i v = _mm512_loadu_si512((const void *)(ilv + 2 * i));
        /* Each 32-bit slot packs [r_hi|r_lo|l_hi|l_lo] (LE), i.e. the
           right sample lives in the high 16 bits, the left in the low
           16 bits.  Arithmetic shifts sign-extend each 16-bit half. */
        __m512i hi = _mm512_srai_epi32(v, 16);                           /* R */
        __m512i lo = _mm512_srai_epi32(_mm512_slli_epi32(v, 16), 16);    /* L */
        /* 16 int32 -> 2 x 8 int32 -> 2 x 8 doubles per channel. */
        __m256i lo_lo = _mm512_castsi512_si256(lo);
        __m256i lo_hi = _mm512_extracti64x4_epi64(lo, 1);
        _mm512_storeu_pd(L + i,      _mm512_mul_pd(sc,
            _mm512_cvtepi32_pd(lo_lo)));
        _mm512_storeu_pd(L + i + 8,  _mm512_mul_pd(sc,
            _mm512_cvtepi32_pd(lo_hi)));
        __m256i hi_lo = _mm512_castsi512_si256(hi);
        __m256i hi_hi = _mm512_extracti64x4_epi64(hi, 1);
        _mm512_storeu_pd(R + i,      _mm512_mul_pd(sc,
            _mm512_cvtepi32_pd(hi_lo)));
        _mm512_storeu_pd(R + i + 8,  _mm512_mul_pd(sc,
            _mm512_cvtepi32_pd(hi_hi)));
    }
    for (; i < n; i++) {
        L[i] = (double)ilv[2 * i + 0] / 32768.0;
        R[i] = (double)ilv[2 * i + 1] / 32768.0;
    }
}

static void fk_deint24(const uint8_t *raw, double *L, double *R, uint32_t n)
{
    /* 24-bit packed samples are awkward to vectorize cheaply; the
       per-sample cost is dominated by the unaligned 3-byte load and
       the sign-extension.  Keeping this scalar matches the AVX2 TU
       (the AVX-512 VBMI byte-shuffle path would only help here and
       isn't worth the extra runtime requirement). */
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *p = raw + 6 * i;
        int32_t l = ((int32_t)p[0] << 8 | (int32_t)p[1] << 16 | (int32_t)p[2] << 24) >> 8;
        int32_t r = ((int32_t)p[3] << 8 | (int32_t)p[4] << 16 | (int32_t)p[5] << 24) >> 8;
        L[i] = (double)l / 8388608.0;
        R[i] = (double)r / 8388608.0;
    }
}

const kf_ops kf_ops_avx512 = {
    fk_analysis, fk_energy_max, fk_threshold, fk_band_max,
    fk_quantize_row, fk_deint16, fk_deint24
};

#endif /* FK_X86 */
