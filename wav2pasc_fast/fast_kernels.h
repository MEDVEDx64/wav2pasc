/*
 * fast_kernels.h — SIMD kernel interface for wav2pasc_fast.
 *
 * Every kernel is bit-identical to the scalar reference arithmetic in
 * wav2pasc.c: identical operation order per output element, separate
 * multiply and add (never contracted into an FMA), IEEE division,
 * round-to-nearest-even, identical integer bit ops.  Vector lanes carry
 * *independent* output elements (different subbands / different k),
 * while the inner reduction order of each element is preserved exactly.
 */

#ifndef FAST_KERNELS_H
#define FAST_KERNELS_H

#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FK_X86 1
#endif

#define FK_NSUB 32
#define FK_NBLK 12
#define FK_XHIST 480

/*
 * ASM level selector — picks one of the kernel implementations below.
 * Used by the runtime dispatch in wav2pasc_fast.c; the WAV2PASC_ASM_LEVEL
 * environment variable takes either the numeric or the symbolic form.
 *
 *   ASM_LEVEL_SCALAR (0) — portable C, bit-exact reference, no SIMD.
 *   ASM_LEVEL_AVX2   (1) — 256-bit AVX2 intrinsics (requires AVX2).
 *   ASM_LEVEL_AVX512 (2) — 512-bit AVX-512F intrinsics (requires AVX-512F).
 */
#define ASM_LEVEL_SCALAR 0
#define ASM_LEVEL_AVX2   1
#define ASM_LEVEL_AVX512 2

/*
 * Analysis contract:
 *   pcmL/pcmR : 384 stereo samples deinterleaved, normalized doubles
 *   xL/xR     : 864 doubles each, xbuf[FK_XHIST + t], t in [-480, 383];
 *               frame at [480..863], retained history at [0..479].
 *               On return [0..479] holds the last 480 samples of the
 *               frame (history for the next frame).  win[idx] of the
 *               original == xbuf[FK_XHIST + 32*b + 31 - idx].
 *   cosT      : transposed cosine matrix cosT[k*32 + m] (same values)
 *   SL/SR     : [block][subband] subband samples
 *
 * Quantize contract: qs = S[i]/scl[i]; if !(qs<1.0) qs = K;
 *   q = omv[i]*qs - off[i]; v = round(q*2^23);
 *   code[i] = (v<<8 ^ 0x80000000) >> 16, zero where amask[i]==0.
 */
typedef struct {
    void (*analysis)(const double *pcmL, const double *pcmR,
                     double *xL, double *xR,
                     const double *coeff, const double *cosT,
                     double *SL, double *SR);
    /* energy e[i] = sum_j S[j*32+i]^2 (j ascending), mx[i] = max_j |S| */
    void (*energy_max)(const double *S, double *e, double *mx);
    /* thr[i] = K1*absM[i] + sum_j spreadT[j*32+i]*e[j] (j ascending) */
    void (*threshold)(const double *e, const double *spreadT,
                      double K1, const double *absM, double *thr);
    void (*band_max)(const double *S, double *mx);
    void (*quantize_row)(const double *S, int32_t n,
                         const double *scl, const double *omv, const double *off,
                         const int32_t *amask, int32_t *code);
    /* interleaved PCM -> two normalized double planes (n = sample pairs) */
    void (*deint16)(const int16_t *ilv, double *L, double *R, uint32_t n);
    void (*deint24)(const uint8_t *raw, double *L, double *R, uint32_t n);
} kf_ops;

extern const kf_ops kf_ops_scalar;
#ifdef FK_X86
extern const kf_ops kf_ops_avx2;
extern const kf_ops kf_ops_avx512;
#endif

#endif
