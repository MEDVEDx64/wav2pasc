/*
 * fast_kernels_scalar.c — reference-order scalar kernels.
 *
 * Every operation order, rounding and constant matches wav2pasc.c
 * bit-for-bit; only the storage layout differs (xbuf history buffer
 * instead of the shifting win[] window, transposed cosT/spreadT —
 * identical values).  Used on CPUs without AVX2 and as the
 * correctness oracle for the AVX2 kernels.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include "fast_kernels.h"

#ifndef FK_XHIST
#define FK_XHIST 480
#endif

static void fk_analysis(const double *pcmL, const double *pcmR,
                 double *xL, double *xR,
                 const double *coeff, const double *cosT,
                 double *SL, double *SR)
{
    double ZL[64], ZR[64];
    int b, i, k, j, m;

    for (i = 0; i < FK_NBLK * 32; i++) {
        xL[FK_XHIST + i] = pcmL[i];
        xR[FK_XHIST + i] = pcmR[i];
    }

    for (b = 0; b < FK_NBLK; b++) {
        /* win[idx] == xbuf[FK_XHIST + 32*b + 31 - idx] in the original */
        const int base = FK_XHIST + 32 * b + 31;
        for (k = 0; k < 64; k++) {
            double al = 0.0, ar = 0.0;
            for (j = 0; j < 8; j++) {
                int idx = 64 * j + k;
                al += xL[base - idx] * coeff[idx];
                ar += xR[base - idx] * coeff[idx];
            }
            ZL[k] = al; ZR[k] = ar;
        }
        for (m = 0; m < FK_NSUB; m++) {
            double sl = 0.0, sr = 0.0;
            for (k = 0; k < 64; k++) {
                sl += cosT[k * FK_NSUB + m] * ZL[k];
                sr += cosT[k * FK_NSUB + m] * ZR[k];
            }
            SL[b * FK_NSUB + m] = sl;
            SR[b * FK_NSUB + m] = sr;
        }
    }

    memmove(xL, xL + FK_NBLK * 32, FK_XHIST * sizeof(double));
    memmove(xR, xR + FK_NBLK * 32, FK_XHIST * sizeof(double));
}

static void fk_energy_max(const double *S, double *e, double *mx)
{
    for (int i = 0; i < FK_NSUB; i++) { e[i] = 0.0; mx[i] = 0.0; }
    for (int j = 0; j < FK_NBLK; j++)
        for (int i = 0; i < FK_NSUB; i++) {
            double sl = S[j * FK_NSUB + i], sr = sl;
            double a = fabs(sl);
            if (a > mx[i]) mx[i] = a;
            e[i] += sl * sr;
        }
}

static void fk_threshold(const double *e, const double *spreadT,
                  double K1, const double *absM, double *thr)
{
    for (int i = 0; i < FK_NSUB; i++) {
        double t = K1 * absM[i];
        for (int j = 0; j < FK_NSUB; j++)
            t += spreadT[j * FK_NSUB + i] * e[j];
        thr[i] = t;
    }
}

static void fk_band_max(const double *S, double *mx)
{
    for (int i = 0; i < FK_NSUB; i++) {
        double m = 0.0;
        for (int j = 0; j < FK_NBLK; j++) {
            double a = fabs(S[j * FK_NSUB + i]);
            if (a > m) m = a;
        }
        mx[i] = m;
    }
}

static long fk_round_double(double x)   /* x87 frndint: nearest even */
{
    double r = nearbyint(x);
    return (long)r;
}

static void fk_quantize_row(const double *S, int32_t n,
                     const double *scl, const double *omv, const double *off,
                     const int32_t *amask, int32_t *code)
{
    for (int i = 0; i < n; i++) {
        if (amask[i]) {
            double qs = S[i] / scl[i];
            if (!(qs < 1.0)) qs = 0.9999998807907104;
            double q = omv[i] * qs - off[i];
            long v = fk_round_double(q * 8388608.0);
            uint32_t u = ((uint32_t)(v & 0xFFFFFFFF) << 8) ^ 0x80000000u;
            code[i] = (int32_t)(u >> 16);
        } else code[i] = 0;
    }
}

static void fk_deint16(const int16_t *ilv, double *L, double *R, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
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

const kf_ops kf_ops_scalar = {
    fk_analysis, fk_energy_max, fk_threshold, fk_band_max,
    fk_quantize_row, fk_deint16, fk_deint24
};
