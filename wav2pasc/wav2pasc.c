/*
 * wav2pasc — PASC (Precision Adaptive Sub-band Coding) / MPEG-1 Layer I encoder
 *
 * Faithful reimplementation of the encoder found in the Win16 application
 * DCC2WAV.EXE ("DCC to Wave Converter", 1996), reverse engineered from its
 * 16-bit Borland Pascal code (segments 2/3 of the NE executable).
 * Output is byte-for-byte identical to the original application.
 *
 * Extensions over the original application:
 *   - 24-bit input WAVs (the original only accepted 8/16-bit input)
 *   - 48000 Hz sample rate (the original only accepted 11.025/22.05/44.1 kHz)
 *   - direct MPEG-1 Layer I elementary stream output (.mp1)
 *
 * Pipeline per frame (384 samples/channel @ 44100/48000 Hz stereo):
 *   1. 512-tap polyphase analysis filter bank (32 subbands x 12 blocks),
 *      prototype window from DCCFIR.COF (coeff = value / 2^21),
 *      cosine matrix M[m][k] = cos((k-16)*(2m+1)*pi/64).
 *   2. Psychoacoustic model: Terhardt absolute threshold, band spreading
 *      matrix (upward exponent 10, downward exponent 27.8),
 *      thr[i] = K1*M[i] + sum_j SP[i][j]*energy[j],
 *      snr[i] = log4(max[i]^2/(3*thr[i])).
 *   3. Bit allocation: gated water-filling on snr
 *      (44.1 kHz: 1520 bits/ch, 48 kHz: 1392 bits/ch;
 *       2-bit activation costs 30 = 2*12 samples + 6-bit scalefactor,
 *       each extra bit costs 12).
 *   4. Scale factors from table 2^(1-i/3) (smallest scale covering max).
 *   5. Quantization: q = (1-2^-n)*clip(S/scale, <1) - 2^-n,
 *      code16 = high word of (round(q*2^23) << 8 XOR sign bit).
 *   6. Frame packing: 32 allocation bytes (nibbles = alloc-1),
 *      MSB-first bit stream of scalefactors (6 bit) and sample codes.
 *
 * Output formats (-f):
 *   0  .mpp PASC container (default): [2-byte rate ID][420-byte records at
 *      44.1 kHz: FF FF C0 00 + 416-byte frame][...]; at 48 kHz: 384-byte
 *      records (FF FF C4 00 + 380-byte frame). This is what DCC-Studio
 *      reads/writes; byte-exact with DCC2WAV.EXE at 44.1/16-bit.
 *   1  .mp1 MPEG-1 Layer I elementary stream: frames of
 *      [FF FF Cx 00 header][payload]. At 44.1 kHz frames are 416 or 420
 *      bytes; the padding bit is set for 24 of every 49 frames (the
 *      DCC-correct pattern, see DCCU/newdccu) so the stream averages
 *      exactly 384 kbps. At 48 kHz all frames are 384 bytes (no padding
 *      needed). Playable by any MPEG-1 audio decoder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "pasc_coeffs.h"

static void setbin_stdin(void)
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
}

static void setbin_stdout(void)
{
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

#define NSUB      32
#define NBLK      12
#define FRAME_SAM (NSUB * NBLK)          /* 384 */
#define BUF_BYTES   0x23C                 /* 572 */
#define MAXSNR      (-10000.0)

enum { FMT_PASC = 0, FMT_MP1 = 1 };

/* ------------------------------------------------------------------ */
/* Rate-dependent parameters                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int      rate;         /* 44100 or 48000                          */
    double   bw;           /* psychoacoustic band width = rate/32     */
    int      frame_bytes;  /* frame payload behind 4-byte header:     */
                           /*   416 (44.1, incl. 4 pad bytes) / 380   */
    int      data_bytes;   /* bytes actually used by the bit budget:  */
                           /*   412 / 380                             */
    int      budget_ch;    /* water-filling bits per channel:         */
                           /*   (frame*8 - 32 hdr - 256 alloc)/2      */
    uint8_t  hdr_byte2;    /* MPEG header byte 2 (rate index)         */
    int      can_pad;      /* padding needed (44.1 kHz only)          */
    uint8_t  mpp_rate_id;  /* .mpp preamble rate ID (44 / 48)         */
} codec_cfg;

static codec_cfg cfg_table[2] = {
    { 44100, 689.0625, 416, 412, 1520, 0xC0, 1, 44 },
    { 48000, 1500.0,   380, 380, 1392, 0xC4, 0, 48 },
};
static codec_cfg cfg;

/* ------------------------------------------------------------------ */
/* TMPEGFrame tables                                                    */
/* ------------------------------------------------------------------ */

static double offsetT[16];   /* offset[n]  = 2^-n          */
static double multT[16];     /* mult[n]    = 2^n/(2^n-1)    */
static double mult2T[16];    /* mult2[n]   = 2*2^-n         */
static double scaleT[64];    /* scaleT[i]  = 2^(1-i/3)      */

/* ------------------------------------------------------------------ */
/* TAllocator psychoacoustic tables                                     */
/* ------------------------------------------------------------------ */

static double absThrM[NSUB];      /* M[i]  = 10^(min(T(flo,fhi))/10)          */
static double spread[NSUB][NSUB]; /* SP[i][j] upward 0.1*(fj/fi)^10,
                                             downward 0.1*(fi/fj)^27.8 */
static const double K1 = 1.0 / 12.0 * 0x1p-35;   /* 2^-35/12  */
static const double K2 = 1.0 / 12.0 * 0x1p-34;   /* 2^-34/12  */

/* ------------------------------------------------------------------ */
/* Filter state                                                         */
/* ------------------------------------------------------------------ */

static double winL[512], winR[512];      /* newest sample at index 0 */
static double coeff[512];                /* COF/2^21                 */
static double cosM[NSUB][64];            /* analysis matrix          */

static double SL[NBLK * NSUB], SR[NBLK * NSUB];   /* subband samples  */

/* allocator working arrays (persist across frames, like the original) */
static uint8_t allocL[NSUB], allocR[NSUB];         /* TAllocator +4/+0x24 */
static uint8_t allocL2[NSUB], allocR2[NSUB];       /* TAllocator +0x44/+0x64 */

static uint8_t dAllocL[NSUB], dAllocR[NSUB];       /* DGROUP 0x33CC/0x33EC */
static uint8_t sfL[NSUB], sfR[NSUB];               /* DGROUP 0x342C/0x340C */
static uint16_t codeL[NBLK * NSUB], codeR[NBLK * NSUB];

/* ------------------------------------------------------------------ */

/* absolute threshold of hearing (Terhardt), dB SPL */
static double thrdb(double f)
{
    double x = f / 1000.0;
    double t;
    if (x < 0.02) x = 0.02;
    if (x == 3.3) t = 0.019115056;
    else t = 3.64 * pow(x, -0.8)
           - 6.5 * exp(-0.6 * (x - 3.3) * (x - 3.3))
           + 0.001 * pow(x, 4.0) + 5.0;
    return t > 60.0 ? 60.0 : t;
}

static void init_tables(void)
{
    int n, i;

    for (n = 2; n <= 15; n++) {
        double p2n = (double)(1 << n);
        offsetT[n] = 1.0 / p2n;
        mult2T[n]  = 2.0 / p2n;
        multT[n]   = 1.0 + 1.0 / (p2n - 1.0);
    }
    for (i = 0; i < 64; i++)
        scaleT[i] = pow(2.0, 1.0 - (double)i / 3.0);

    for (i = 0; i < 512; i++)
        coeff[i] = (double)pasc_cof[i] / 2097152.0;   /* /2^21 */

    for (int m = 0; m < NSUB; m++)
        for (int k = 0; k < 64; k++)
            cosM[m][k] = cos((k - 16) * (2 * m + 1) * M_PI / 64.0);

    for (i = 0; i < NSUB; i++) {
        double flo = (i == 0) ? 20.0 : i * cfg.bw;
        double fhi = (i + 1) * cfg.bw;
        double t = thrdb(flo), t2 = thrdb(fhi);
        if (t2 < t) t = t2;                       /* min of band edges */
        absThrM[i] = pow(10.0, t / 10.0);
    }

    /* spreading matrix; upward loop includes c == r-1 (original quirk) */
    const double bw = cfg.bw;
    for (int r = 0; r < NSUB; r++) {
        double f_hi = (r + 1) * bw;
        for (int c = 0; c <= r - 1; c++) {
            double f_j = (c == 0) ? 20.0 : c * bw;
            spread[r][c] = 0.1 * pow(f_j / f_hi, 10.0);
        }
        double f_lo = (r == 0) ? 20.0 : r * bw;
        for (int c = r; c < NSUB; c++) {
            double f_jhi = (c + 1) * bw;
            spread[r][c] = 0.1 * pow(f_lo / f_jhi, 27.8);
        }
    }

    memset(allocL, 0, sizeof allocL);
    memset(allocR, 0, sizeof allocR);
    memset(allocL2, 0, sizeof allocL2);
    memset(allocR2, 0, sizeof allocR2);
    for (i = 0; i < NSUB; i++) { allocL2[i] = 5; allocR2[i] = 5; }
    memset(winL, 0, sizeof winL);
    memset(winR, 0, sizeof winR);
}

/* ------------------------------------------------------------------ */
/* Polyphase analysis: one block of 32 stereo samples.                  */
/* Input is normalized doubles in [-1, 1); for 16-bit WAVs this is      */
/* identical to the original's int32 = s16<<2 then /2^17 path.          */
/* base = block index 0..11                                             */
/* ------------------------------------------------------------------ */

static void analysis_block(const double *pcm, int base)
{
    double ZL[64], ZR[64];
    int i, k, j, m;

    /* shift window: W[i] = W[i-32], oldest 32 discarded */
    for (i = 511; i >= 32; i--) {
        winL[i] = winL[i - 32];
        winR[i] = winR[i - 32];
    }

    for (i = 0; i < 32; i++) {
        winL[31 - i] = pcm[(base * 32 + i) * 2 + 0];
        winR[31 - i] = pcm[(base * 32 + i) * 2 + 1];
    }

    for (k = 0; k < 64; k++) {
        double al = 0.0, ar = 0.0;
        for (j = 0; j < 8; j++) {
            int idx = 64 * j + k;
            al += winL[idx] * coeff[idx];
            ar += winR[idx] * coeff[idx];
        }
        ZL[k] = al; ZR[k] = ar;
    }
    for (m = 0; m < NSUB; m++) {
        double sl = 0.0, sr = 0.0;
        const double *cm = cosM[m];
        for (k = 0; k < 64; k++) {
            sl += cm[k] * ZL[k];
            sr += cm[k] * ZR[k];
        }
        SL[base * NSUB + m] = sl;
        SR[base * NSUB + m] = sr;
    }
}

/* ------------------------------------------------------------------ */
/* Adaptive bit allocation (seg02:0x105E)                               */
/* ------------------------------------------------------------------ */

static void allocate_bits(void)
{
    double eL[NSUB], eR[NSUB], mxL[NSUB], mxR[NSUB];
    double thrL[NSUB], thrR[NSUB], snrL[NSUB], snrR[NSUB];
    int i, j;

    for (i = 0; i < NSUB; i++) {
        dAllocL[i] = 0; dAllocR[i] = 0;
        eL[i] = eR[i] = 0.0; mxL[i] = mxR[i] = 0.0;
    }

    for (j = 0; j < NBLK; j++)
        for (i = 0; i < NSUB; i++) {
            double sl = SL[j * NSUB + i], sr = SR[j * NSUB + i];
            double al = fabs(sl), ar = fabs(sr);
            if (al > mxL[i]) mxL[i] = al;
            if (ar > mxR[i]) mxR[i] = ar;
            eL[i] += sl * sl;
            eR[i] += sr * sr;
        }
    for (i = 0; i < NSUB; i++)
        eL[i] /= 12.0;    /* note: original divides only the L energies */

    for (i = 0; i < NSUB; i++) {
        double tl = K1 * absThrM[i], tr = K1 * absThrM[i];
        for (j = 0; j < NSUB; j++) {
            tl += spread[i][j] * eL[j];
            tr += spread[i][j] * eR[j];
        }
        thrL[i] = tl; thrR[i] = tr;
    }

    for (i = 0; i < NSUB; i++) {
        /* snr = log4(max^2/(3*thr)) = ln(ratio)/(2*ln2)   (seg02:0x13E7..0x144D) */
        if (thrL[i] > 0.0 && mxL[i] * mxL[i] > 0.0)
            snrL[i] = log(mxL[i] * mxL[i] / (3.0 * thrL[i])) / (2.0 * M_LN2);
        else
            snrL[i] = MAXSNR;
        if (thrR[i] > 0.0 && mxR[i] * mxR[i] > 0.0)
            snrR[i] = log(mxR[i] * mxR[i] / (3.0 * thrR[i])) / (2.0 * M_LN2);
        else
            snrR[i] = MAXSNR;
    }

    /* pass 4: gate on energy vs threshold */
    for (i = 0; i < NSUB; i++) {
        if (eL[i] >= thrL[i]) allocL[i] = 5;
        else if (allocL[i] > 0) allocL[i]--;
        if (eR[i] >= thrR[i]) allocR[i] = 5;
        else if (allocR[i] > 0) allocR[i]--;
    }
    /* pass 5: stereo max (seg02:0x1645: jae->allocR=allocL else allocL=allocR) */
    for (i = 0; i < NSUB; i++) {
        uint8_t m = allocL[i] > allocR[i] ? allocL[i] : allocR[i];
        allocL[i] = m; allocR[i] = m;
    }
    /* pass 6 */
    for (i = 0; i < NSUB; i++) {
        if (allocL[i] == 5) allocL2[i] = 5;
        else if (eL[i] >= K2) { if (allocL2[i] > 0) allocL2[i]--; }
        else if (allocL[i] == 0) allocL2[i] = 5;
        if (allocR[i] == 5) allocR2[i] = 5;
        else if (eR[i] >= K2) { if (allocR2[i] > 0) allocR2[i]--; }
        else if (allocR[i] == 0) allocR2[i] = 5;
    }

    /* water-filling per channel */
    for (int ch = 0; ch < 2; ch++) {
        uint8_t *dg  = ch ? dAllocR : dAllocL;
        uint8_t *al  = ch ? allocR  : allocL;
        uint8_t *al2 = ch ? allocR2 : allocL2;
        double *snr  = ch ? snrR    : snrL;
        int budget = cfg.budget_ch;

        for (i = 0; i < NSUB; i++) {
            if (al[i] > 0) {
                dg[i] = 2;
                snr[i] -= 2.0;      /* seg02:0x1853: fsub dword cs:[0x1051] (=2.0) */
                budget -= 30;
            }
        }
        for (;;) {
            int best = 0;
            double bv = snr[0];
            for (i = 1; i < NSUB; i++)
                if (bv < snr[i]) { best = i; bv = snr[i]; }
            if (bv == MAXSNR) break;

            int step = dg[best] == 0 ? 2 : 1;
            int cost = dg[best] == 0 ? 30 : 12;
            int invalidated = 0;
            if (cost > budget || dg[best] == 15)
                invalidated = 1;
            else if (al2[best] > 0 && al[best] == 0)
                invalidated = 1;    /* original inverts these gates */
            if (invalidated) snr[best] = MAXSNR;

            if (snr[best] != MAXSNR) {
                dg[best] = (uint8_t)(dg[best] + step);
                snr[best] -= step;
                budget -= cost;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scale factors (seg02:0x6B7)                                          */
/* ------------------------------------------------------------------ */

static void compute_scale_factors(void)
{
    for (int i = 0; i < NSUB; i++) {
        int sf; double mx;
        mx = 0.0;
        for (int j = 0; j < NBLK; j++) { double a = fabs(SL[j*NSUB+i]); if (a > mx) mx = a; }
        sf = 62;
        while (!(mx < scaleT[sf])) { if (sf <= 0) break; sf--; }
        sfL[i] = (uint8_t)sf;
        mx = 0.0;
        for (int j = 0; j < NBLK; j++) { double a = fabs(SR[j*NSUB+i]); if (a > mx) mx = a; }
        sf = 62;
        while (!(mx < scaleT[sf])) { if (sf <= 0) break; sf--; }
        sfR[i] = (uint8_t)sf;
    }
}

/* ------------------------------------------------------------------ */
/* Quantizer (seg02:0x832)                                              */
/* ------------------------------------------------------------------ */

static long round_double(double x)    /* x87 frndint: nearest even */
{
    double r = nearbyint(x);
    return (long)r;
}

static void quantize(void)
{
    for (int j = 0; j < NBLK; j++)
        for (int i = 0; i < NSUB; i++) {
            int idx = j * NSUB + i;
            /* left */
            if (dAllocL[i] > 0) {
                int n = dAllocL[i];
                double qs = SL[idx] / scaleT[sfL[i]];
                if (!(qs < 1.0)) qs = 0.9999998807907104;
                double q = (1.0 - offsetT[n]) * qs - offsetT[n];
                long v = round_double(q * 8388608.0);   /* 2^23 */
                uint32_t u = ((uint32_t)(v & 0xFFFFFFFF) << 8) ^ 0x80000000u;
                codeL[idx] = (uint16_t)(u >> 16);
            } else codeL[idx] = 0;
            /* right */
            if (dAllocR[i] > 0) {
                int n = dAllocR[i];
                double qs = SR[idx] / scaleT[sfR[i]];
                if (!(qs < 1.0)) qs = 0.9999998807907104;
                double q = (1.0 - offsetT[n]) * qs - offsetT[n];
                long v = round_double(q * 8388608.0);
                uint32_t u = ((uint32_t)(v & 0xFFFFFFFF) << 8) ^ 0x80000000u;
                codeR[idx] = (uint16_t)(u >> 16);
            } else codeR[idx] = 0;
        }
}

/* ------------------------------------------------------------------ */
/* Frame packing (seg02:0x3447/0x3597/0x3607)                           */
/* ------------------------------------------------------------------ */

static uint8_t fbuf[BUF_BYTES];
static int bitpos;

static void put_bits(uint16_t val, int count)
{
    /* faithful MSB-first writer: bit b uses the original's fixed mask table
       W[p] = 1 << (16-p), i.e. the TOP `count` bits of the 16-bit value */
    for (int b = 1; b <= count; b++) {
        if ((bitpos & 7) == 0)
            fbuf[bitpos >> 3] = 0;
        if (val & (1 << (16 - b)))
            fbuf[bitpos >> 3] |= 0x80 >> (bitpos & 7);
        bitpos++;
    }
}

static int build_frame(void)
{
    memset(fbuf, 0, sizeof fbuf);

    for (int i = 0; i < NSUB; i++) {
        uint8_t b = 0;
        if (dAllocL[i] > 0) b |= (uint8_t)((dAllocL[i] - 1) << 4);
        if (dAllocR[i] > 0) b |= (uint8_t)(dAllocR[i] - 1);
        fbuf[i] = b;
    }
    bitpos = 0x100;

    for (int i = 0; i < NSUB; i++) {
        if (dAllocL[i] > 0) put_bits((uint16_t)(sfL[i] << 10), 6);
        if (dAllocR[i] > 0) put_bits((uint16_t)(sfR[i] << 10), 6);
    }

    for (int j = 0; j < NBLK; j++)
        for (int i = 0; i < NSUB; i++) {
            if (dAllocL[i] > 0) put_bits(codeL[j * NSUB + i], dAllocL[i]);
            if (dAllocR[i] > 0) put_bits(codeR[j * NSUB + i], dAllocR[i]);
        }

    /* trailing marker byte (only if it fits inside the frame;
       at 44.1 kHz it always does, at 48 kHz the budget fills the frame) */
    if (bitpos + 8 <= cfg.frame_bytes * 8)
        put_bits(0, 8);
    return cfg.frame_bytes;
}

/* ------------------------------------------------------------------ */
/* WAV reading (16/24-bit PCM incl. WAVE_FORMAT_EXTENSIBLE)            */
/* Sequential-only, one frame at a time: the input may be a pipe ('-') */
/* and is never held in memory in its entirety.                        */
/* ------------------------------------------------------------------ */

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1]<<8 | p[2]<<16 | (uint32_t)p[3]<<24; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1]<<8); }

static void skip_bytes(FILE *f, long n)
{
    uint8_t scratch[4096];
    while (n > 0) {
        size_t c = (size_t)n < sizeof scratch ? (size_t)n : sizeof scratch;
        if (fread(scratch, 1, c, f) != c) break;
        n -= (long)c;
    }
}

static struct {
    FILE    *f;
    int      bits;       /* 16 or 24                              */
    int      sbytes;     /* bytes per sample (2 or 3)             */
    uint32_t datalen;    /* data chunk size; 0xFFFFFFFF = unknown */
    uint32_t consumed;   /* bytes read so far                     */
} wav;

static uint8_t frame_raw[FRAME_SAM * 2 * 3];   /* one frame, 24-bit worst case */
static double frame_pcm[FRAME_SAM * 2];        /* normalized interleaved stereo */

/* parse + validate the header; leaves the stream at the data chunk */
static int open_wav(FILE *f, const char *name, uint32_t *rate_out)
{
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "%s: not a RIFF/WAVE file\n", name);
        return -1;
    }
    uint16_t nch = 0, bits = 0, fmt_tag = 0;
    uint32_t rate = 0, datalen = 0;
    int have_fmt = 0;
    uint8_t chunk[8];
    while (fread(chunk, 1, 8, f) == 8) {
        uint32_t sz = rd32(chunk + 4);
        if (!memcmp(chunk, "fmt ", 4)) {
            uint8_t fmt[64];
            uint32_t r = sz < 64 ? sz : 64;
            if (fread(fmt, 1, r, f) != r) break;
            if (r < 16) { fprintf(stderr, "%s: bad fmt chunk\n", name); return -1; }
            fmt_tag = rd16(fmt);
            if (fmt_tag == 0xFFFE) {          /* WAVE_FORMAT_EXTENSIBLE */
                if (r < 40 || rd16(fmt + 16) < 22) {
                    fprintf(stderr, "%s: bad extensible fmt chunk\n", name);
                    return -1;
                }
                fmt_tag = rd16(fmt + 24);     /* SubFormat GUID first word */
            }
            nch = rd16(fmt + 2); rate = rd32(fmt + 4); bits = rd16(fmt + 14);
            have_fmt = 1;
            if (sz > r) skip_bytes(f, (long)(sz - r + (sz & 1)));
        } else if (!memcmp(chunk, "data", 4)) {
            datalen = sz;
            break;
        } else {
            skip_bytes(f, (long)(sz + (sz & 1)));
        }
    }
    if (!have_fmt) {
        fprintf(stderr, "%s: no fmt chunk found\n", name);
        return -1;
    }
    if (fmt_tag != 1) {
        fprintf(stderr, "unsupported WAV format tag %u (PCM only)\n", fmt_tag);
        return -1;
    }
    if (nch != 2) {
        fprintf(stderr, "input must be stereo (got %u channels)\n", nch);
        return -1;
    }
    if (rate != 44100 && rate != 48000) {
        fprintf(stderr, "input sample rate must be 44100 or 48000 Hz (got %u)\n"
                        "resample first, e.g.: sox in.wav -r 44100 out.wav\n", rate);
        return -1;
    }
    if (bits != 16 && bits != 24) {
        fprintf(stderr, "unsupported sample format: %u bits (16 or 24 supported)\n", bits);
        return -1;
    }

    wav.f = f;
    wav.bits = bits;
    wav.sbytes = bits / 8;
    wav.datalen = datalen;
    wav.consumed = 0;
    *rate_out = rate;
    return 0;
}

/* read the next frame; returns stereo pairs read (0..FRAME_SAM) as
   normalized interleaved doubles in [-1, 1) in frame_pcm */
static int read_pcm_frame(void)
{
    size_t want = FRAME_SAM * 2 * (size_t)wav.sbytes;
    if (wav.datalen != 0xFFFFFFFFu && wav.datalen - wav.consumed < want)
        want = wav.datalen - wav.consumed;
    size_t got = fread(frame_raw, 1, want, wav.f);
    wav.consumed += (uint32_t)got;
    uint32_t nfr = (uint32_t)(got / (2u * (size_t)wav.sbytes));
    if (wav.bits == 16) {
        for (uint32_t i = 0; i < nfr * 2; i++)
            frame_pcm[i] = (double)(int16_t)rd16(frame_raw + 2 * i) / 32768.0;
    } else {
        for (uint32_t i = 0; i < nfr * 2; i++) {
            const uint8_t *p = frame_raw + 3 * i;
            int32_t v = ((int32_t)p[0] << 8 | (int32_t)p[1] << 16 | (int32_t)p[2] << 24) >> 8;
            frame_pcm[i] = (double)v / 8388608.0;
        }
    }
    return (int)nfr;
}

/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    printf("usage: %s [options] input.wav output.mpp\n"
           "\n"
           "PASC / MPEG-1 Layer I encoder (DCC2WAV.EXE algorithm reimplementation).\n"
           "\n"
           "options:\n"
           "  -f, --format 0|1   output format (default 0)\n"
           "                       0 = PASC .mpp container (DCC-Studio compatible)\n"
           "                       1 = MPEG-1 Layer I .mp1 elementary stream\n"
           "  -h, --help         this help\n"
           "\n"
           "input:  44100 or 48000 Hz stereo PCM WAV, 16 or 24-bit\n"
           "  (24-bit input exploits the codec's full quantization range;\n"
           "   the original DCC2WAV.EXE accepted 16-bit only)\n"
           "  use '-' as the file name to read from stdin / write to stdout\n"
           "output: -f 0: [rate-ID 2C 00][420-byte records at 44.1 kHz:\n"
           "           FF FF C0 00 + 416-byte frame][...]\n"
           "         -f 1: MPEG frames with DCC padding pattern at 44.1 kHz\n"
           "           (24 of every 49 frames padded), 384-byte frames at 48 kHz\n",
           prog);
}

int main(int argc, char **argv)
{
    const char *inpath = NULL, *outpath = NULL;
    int fmt = FMT_PASC;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        }
        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
            if (i + 1 >= argc) { fprintf(stderr, "-f needs an argument\n"); return 1; }
            fmt = atoi(argv[++i]);
            if (fmt != 0 && fmt != 1) { fprintf(stderr, "-f must be 0 or 1\n"); return 1; }
            continue;
        }
        if (argv[i][0] == '-' && argv[i][1] != '\0') {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
        if (!inpath) inpath = argv[i];
        else if (!outpath) outpath = argv[i];
        else { fprintf(stderr, "too many arguments\n"); return 1; }
    }
    if (!inpath || !outpath) {
        usage(argv[0]);
        return 1;
    }

    uint32_t rate = 0;

    FILE *in = stdin;
    if (strcmp(inpath, "-")) {
        in = fopen(inpath, "rb");
        if (!in) { perror(inpath); return 1; }
    } else {
        setbin_stdin();
    }
    if (open_wav(in, inpath, &rate) != 0) {
        if (in != stdin) fclose(in);
        return 1;
    }

    cfg = (rate == 44100) ? cfg_table[0] : cfg_table[1];

    if (rate == 48000 && fmt == FMT_PASC)
        fprintf(stderr, "warning: 48 kHz PASC (.mpp) is not playable on DCC decks; "
                        "consider -f 1 for an MPEG-1 Layer I stream\n");

    FILE *out = stdout;
    if (strcmp(outpath, "-")) {
        out = fopen(outpath, "wb");
        if (!out) { perror(outpath); return 1; }
    } else {
        setbin_stdout();
    }

    if (fmt == FMT_PASC) {
        /* .mpp preamble: rate ID (44 = 44100, 48 = 48000) + 0x00 */
        uint8_t pre[2] = { cfg.mpp_rate_id, 0x00 };
        fwrite(pre, 1, 2, out);
    }

    init_tables();

    /* original drops the trailing partial frame; with a known data size
       the frame total is known up front for the progress display */
    int known_len = wav.datalen != 0xFFFFFFFFu;
    uint32_t nframes = known_len
        ? wav.datalen / (2u * (uint32_t)wav.sbytes) / FRAME_SAM : 0;

    /* .mp1 padding state: exact newdccu/DCC pattern. newdccu's packet
       counter i starts at 1 (0 is consumed by sync search); frames with
       even i are padded (c2/420), and after every 49th written frame an
       extra increment keeps 24 padded + 25 unpadded per group (average
       exactly 384 kbps at 44.1 kHz). */
    unsigned ndcc_i = 1;

    for (uint32_t fr = 0; ; fr++) {
        if (read_pcm_frame() < FRAME_SAM) break;   /* EOF or partial frame */
        const double *framepcm = frame_pcm;

        for (int b = 0; b < NBLK; b++)
            analysis_block(framepcm, b);

        allocate_bits();
        compute_scale_factors();
        quantize();
        int len = build_frame();

        if (fmt == FMT_MP1) {
            int padded = cfg.can_pad ? (ndcc_i % 2 == 0) : 0;
            uint8_t hdr[4] = {0xFF, 0xFF, (uint8_t)(cfg.hdr_byte2 | (padded ? 2 : 0)), 0x00};
            fwrite(hdr, 1, 4, out);
            fwrite(fbuf, 1, (size_t)cfg.data_bytes + (padded ? 4 : 0), out);
            if (cfg.can_pad) {
                ndcc_i++;
                if ((fr + 1) % 49 == 0) ndcc_i++;   /* packets>49 rollover */
            }
        } else {
            /* .mpp record: 4-byte sync + full frame (padded to 416 at 44.1) */
            uint8_t hdr[4] = {0xFF, 0xFF, cfg.hdr_byte2, 0x00};
            fwrite(hdr, 1, 4, out);
            fwrite(fbuf, 1, (size_t)len, out);
        }

        if ((fr & 31) == 0) {
            if (known_len) fprintf(stderr, "\rframe %u/%u", fr + 1, nframes);
            else           fprintf(stderr, "\rframe %u", fr + 1);
            fflush(stderr);
        }
    }
    if (in != stdin) fclose(in);
    if (known_len && wav.consumed < wav.datalen)
        fprintf(stderr, "%s: warning: data chunk truncated (%u of %u bytes)\n",
                inpath, wav.consumed, wav.datalen);
    fputc('\n', stderr);
    if (out != stdout) fclose(out);
    else fflush(out);
    return 0;
}