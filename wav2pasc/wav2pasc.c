/*
 * wav2pasc — PASC / MPEG-1 Layer I encoder (DCC2WAV.EXE algorithm
 * reimplementation), with the hot DSP path (filter bank, energies,
 * thresholds, quantization, PCM deinterleaving) written as hand-crafted
 * SIMD kernels: AVX-512F and AVX2 intrinsics with a bit-identical
 * scalar fallback (fast_kernels_avx512.c, fast_kernels_avx2.c,
 * fast_kernels_scalar.c) that keeps the app correct and byte-exact on
 * CPUs without AVX2/AVX-512 and on non-x86 targets.
 *
 * Runtime dispatch (see pick_asm_level() below):
 *   WAV2PASC_ASM_LEVEL=0|scalar   force scalar
 *   WAV2PASC_ASM_LEVEL=1|avx2     force AVX2 (if CPU supports it)
 *   WAV2PASC_ASM_LEVEL=2|avx512   force AVX-512 (if CPU supports it)
 *   unset                         pick the highest level the CPU supports
 *
 * Byte-exactness rules enforced here:
 *   - identical table values (same expressions, same order);
 *   - per-element operation order preserved inside every kernel
 *     (vector lanes carry independent elements; reductions keep the
 *     original sequential order);
 *   - separate multiply and add, never FMA; the AVX2 and AVX-512 TUs
 *     are compiled with -ffp-contract=off;
 *   - IEEE division, round-to-nearest-even, same integer bit-twiddling;
 *   - the bit-serial put_bits of the original replaced by an equivalent
 *     MSB-first byte writer producing identical frame bytes.
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
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include "pasc_coeffs.h"
#include "fast_kernels.h"

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

typedef struct {
    int      rate;
    double   bw;
    int      frame_bytes;
    int      data_bytes;
    int      budget_ch;
    uint8_t  hdr_byte2;
    int      can_pad;
    uint8_t  mpp_rate_id;
} codec_cfg;

static codec_cfg cfg_table[2] = {
    { 44100, 689.0625, 416, 412, 1520, 0xC0, 1, 44 },
    { 48000, 1500.0,   380, 380, 1392, 0xC4, 0, 48 },
};
static codec_cfg cfg;

static double offsetT[16];
static double multT[16];
static double mult2T[16];
static double scaleT[64];

static double absThrM[NSUB];
static double spread[NSUB][NSUB];      /* kept in original [i][j] order   */
static double spreadT[NSUB * NSUB];    /* transposed: spreadT[j*32 + i]   */
static double cosT[64 * NSUB];         /* cosT[k*32 + m] (transposed)     */
static const double K1 = 1.0 / 12.0 * 0x1p-35;
static const double K2 = 1.0 / 12.0 * 0x1p-34;

/* signal history + extended frame buffer (see kf_ops.analysis contract) */
#define XLEN   (FK_XHIST + FRAME_SAM)   /* 864 */
static double xbufL[XLEN], xbufR[XLEN];
static double coeff[512];

static double SL[NBLK * NSUB], SR[NBLK * NSUB];   /* subband samples  */

static uint8_t allocL[NSUB], allocR[NSUB];
static uint8_t allocL2[NSUB], allocR2[NSUB];

static uint8_t dAllocL[NSUB], dAllocR[NSUB];
static uint8_t sfL[NSUB], sfR[NSUB];
static double sclL[NSUB], omvL[NSUB], offL[NSUB], sclR[NSUB], omvR[NSUB], offR[NSUB];
static int32_t amL[NSUB], amR[NSUB];
static int32_t codeL[NBLK * NSUB], codeR[NBLK * NSUB];

/* ------------------------------------------------------------------ */
/* kernel dispatch                                                      */
/* ------------------------------------------------------------------ */

static const kf_ops *K = &kf_ops_scalar;

#ifdef FK_X86
static int avx2_available(void)
{
#ifdef __GNUC__
    return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 1);
    if (!(regs[2] & (1 << 27))) return 0;         /* OSXSAVE  */
    if (!(regs[2] & (1 << 28))) return 0;         /* AVX      */
    if ((_xgetbv(0) & 0x6) != 0x6) return 0;      /* XMM+YMM state */
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;             /* AVX2     */
#else
    return 0;
#endif
}

/* AVX-512F detection: requires OSXSAVE, AVX, the OS to enable the
   opmask/ZMM state in XCR0 (bits 5..7), and the AVX-512F feature bit
   in CPUID leaf 7 sub-leaf 0 EBX[16].  Only AVX-512F is required by
   the kernels in fast_kernels_avx512.c. */
static int avx512_available(void)
{
#ifdef __GNUC__
    return __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 1);
    if (!(regs[2] & (1 << 27))) return 0;         /* OSXSAVE  */
    if (!(regs[2] & (1 << 28))) return 0;         /* AVX      */
    if ((_xgetbv(0) & 0xE6) != 0xE6) return 0;    /* XMM+YMM+opmask+ZMM_Hi256+Hi16_ZMM */
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 16)) != 0;            /* AVX-512F */
#else
    return 0;
#endif
}
#endif /* FK_X86 */

/*
 * Pick the kernel set:
 *   - if WAV2PASC_ASM_LEVEL is set, it forces a level (numeric 0/1/2
 *     or symbolic "scalar"/"avx2"/"avx512"); a level above what the
 *     CPU supports is silently clipped down to the highest available;
 *   - otherwise the highest level supported by the CPU is used.
 *
 * Replaces the old WAV2PASC_SCALAR flag from the now-removed reference encoder.
 */
static int pick_asm_level(void)
{
    int level = ASM_LEVEL_SCALAR;
#ifdef FK_X86
    if (avx2_available())   level = ASM_LEVEL_AVX2;
    if (avx512_available()) level = ASM_LEVEL_AVX512;
#endif
    const char *env = getenv("WAV2PASC_ASM_LEVEL");
    if (env) {
        int req = -1;
        if      (!strcmp(env, "0") || !strcmp(env, "scalar")) req = ASM_LEVEL_SCALAR;
        else if (!strcmp(env, "1") || !strcmp(env, "avx2"))   req = ASM_LEVEL_AVX2;
        else if (!strcmp(env, "2") || !strcmp(env, "avx512")) req = ASM_LEVEL_AVX512;
        if (req >= 0 && req < level) level = req;
    }
    return level;
}

static const char *asm_level_name(int level)
{
    switch (level) {
        case ASM_LEVEL_AVX512: return "avx512";
        case ASM_LEVEL_AVX2:   return "avx2";
        default:               return "scalar";
    }
}

/* absolute threshold of hearing (Terhardt), dB SPL — as in wav2pasc.c */
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

    /* identical expressions as the original, transposed placement */
    for (int m = 0; m < NSUB; m++)
        for (int k = 0; k < 64; k++)
            cosT[k * NSUB + m] = cos((k - 16) * (2 * m + 1) * M_PI / 64.0);

    for (i = 0; i < NSUB; i++) {
        double flo = (i == 0) ? 20.0 : i * cfg.bw;
        double fhi = (i + 1) * cfg.bw;
        double t = thrdb(flo), t2 = thrdb(fhi);
        if (t2 < t) t = t2;
        absThrM[i] = pow(10.0, t / 10.0);
    }

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
    for (int r = 0; r < NSUB; r++)
        for (int c = 0; c < NSUB; c++)
            spreadT[c * NSUB + r] = spread[r][c];

    memset(allocL, 0, sizeof allocL);
    memset(allocR, 0, sizeof allocR);
    memset(allocL2, 0, sizeof allocL2);
    memset(allocR2, 0, sizeof allocR2);
    for (i = 0; i < NSUB; i++) { allocL2[i] = 5; allocR2[i] = 5; }
    memset(xbufL, 0, sizeof xbufL);
    memset(xbufR, 0, sizeof xbufR);
}

/* ------------------------------------------------------------------ */

static void allocate_bits(void)
{
    double eL[NSUB], eR[NSUB], mxL[NSUB], mxR[NSUB];
    double thrL[NSUB], thrR[NSUB], snrL[NSUB], snrR[NSUB];
    int i;

    for (i = 0; i < NSUB; i++) {
        dAllocL[i] = 0; dAllocR[i] = 0;
    }

    K->energy_max(SL, eL, mxL);
    K->energy_max(SR, eR, mxR);
    for (i = 0; i < NSUB; i++)
        eL[i] /= 12.0;    /* note: original divides only the L energies */

    K->threshold(eL, spreadT, K1, absThrM, thrL);
    K->threshold(eR, spreadT, K1, absThrM, thrR);

    for (i = 0; i < NSUB; i++) {
        if (thrL[i] > 0.0 && mxL[i] * mxL[i] > 0.0)
            snrL[i] = log(mxL[i] * mxL[i] / (3.0 * thrL[i])) / (2.0 * M_LN2);
        else
            snrL[i] = MAXSNR;
        if (thrR[i] > 0.0 && mxR[i] * mxR[i] > 0.0)
            snrR[i] = log(mxR[i] * mxR[i] / (3.0 * thrR[i])) / (2.0 * M_LN2);
        else
            snrR[i] = MAXSNR;
    }

    for (i = 0; i < NSUB; i++) {
        if (eL[i] >= thrL[i]) allocL[i] = 5;
        else if (allocL[i] > 0) allocL[i]--;
        if (eR[i] >= thrR[i]) allocR[i] = 5;
        else if (allocR[i] > 0) allocR[i]--;
    }
    for (i = 0; i < NSUB; i++) {
        uint8_t m = allocL[i] > allocR[i] ? allocL[i] : allocR[i];
        allocL[i] = m; allocR[i] = m;
    }
    for (i = 0; i < NSUB; i++) {
        if (allocL[i] == 5) allocL2[i] = 5;
        else if (eL[i] >= K2) { if (allocL2[i] > 0) allocL2[i]--; }
        else if (allocL[i] == 0) allocL2[i] = 5;
        if (allocR[i] == 5) allocR2[i] = 5;
        else if (eR[i] >= K2) { if (allocR2[i] > 0) allocR2[i]--; }
        else if (allocR[i] == 0) allocR2[i] = 5;
    }

    for (int ch = 0; ch < 2; ch++) {
        uint8_t *dg  = ch ? dAllocR : dAllocL;
        uint8_t *al  = ch ? allocR  : allocL;
        uint8_t *al2 = ch ? allocR2 : allocL2;
        double *snr  = ch ? snrR    : snrL;
        int budget = cfg.budget_ch;

        for (i = 0; i < NSUB; i++) {
            if (al[i] > 0) {
                dg[i] = 2;
                snr[i] -= 2.0;
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
                invalidated = 1;
            if (invalidated) snr[best] = MAXSNR;

            if (snr[best] != MAXSNR) {
                dg[best] = (uint8_t)(dg[best] + step);
                snr[best] -= step;
                budget -= cost;
            }
        }
    }
}

static void compute_scale_factors(void)
{
    double mxL[NSUB], mxR[NSUB];
    K->band_max(SL, mxL);
    K->band_max(SR, mxR);
    for (int i = 0; i < NSUB; i++) {
        int sf = 62;
        double mx = mxL[i];
        while (!(mx < scaleT[sf])) { if (sf <= 0) break; sf--; }
        sfL[i] = (uint8_t)sf;
        sf = 62;
        mx = mxR[i];
        while (!(mx < scaleT[sf])) { if (sf <= 0) break; sf--; }
        sfR[i] = (uint8_t)sf;
    }
}

static void quantize(void)
{
    for (int i = 0; i < NSUB; i++) {
        int n = dAllocL[i];
        if (n > 0) {
            sclL[i] = scaleT[sfL[i]];
            offL[i] = offsetT[n];
            omvL[i] = 1.0 - offsetT[n];
            amL[i] = -1;
        } else {
            sclL[i] = 1.0; omvL[i] = 0.0; offL[i] = 0.0; amL[i] = 0;
        }
        n = dAllocR[i];
        if (n > 0) {
            sclR[i] = scaleT[sfR[i]];
            offR[i] = offsetT[n];
            omvR[i] = 1.0 - offsetT[n];
            amR[i] = -1;
        } else {
            sclR[i] = 1.0; omvR[i] = 0.0; offR[i] = 0.0; amR[i] = 0;
        }
    }
    for (int j = 0; j < NBLK; j++) {
        K->quantize_row(SL + j * NSUB, NSUB, sclL, omvL, offL, amL, codeL + j * NSUB);
        K->quantize_row(SR + j * NSUB, NSUB, sclR, omvR, offR, amR, codeR + j * NSUB);
    }
}

/* ------------------------------------------------------------------ */
/* Frame packing — identical bytes to the original bit-serial writer.   */
/* put_bits appends the TOP `count` bits of `val`, MSB first; since the */
/* stream starts byte-aligned (bit 0x100) and the buffer is zeroed, an  */
/* MSB-first 64-bit accumulator writer yields the same bytes.          */
/* ------------------------------------------------------------------ */

static uint8_t fbuf[BUF_BYTES];
static int bitpos;
static uint64_t bacc;
static int nb;
static uint8_t *bout;

static void put_bits(uint16_t val, int count)
{
    uint64_t vb = (uint64_t)(val >> (16 - count));   /* the top `count` bits */
    int room = 64 - nb;
    if (count <= room) {
        bacc |= vb << (room - count);
        nb += count;
    } else {
        int spill = count - room;
        bacc |= vb >> spill;
        *bout++ = (uint8_t)(bacc >> 56);
        *bout++ = (uint8_t)(bacc >> 48);
        *bout++ = (uint8_t)(bacc >> 40);
        *bout++ = (uint8_t)(bacc >> 32);
        *bout++ = (uint8_t)(bacc >> 24);
        *bout++ = (uint8_t)(bacc >> 16);
        *bout++ = (uint8_t)(bacc >> 8);
        *bout++ = (uint8_t)bacc;
        bacc = vb << (64 - spill);
        nb = spill;
    }
    bitpos += count;
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
    bacc = 0; nb = 0; bout = fbuf + (0x100 >> 3);

    for (int i = 0; i < NSUB; i++) {
        if (dAllocL[i] > 0) put_bits((uint16_t)(sfL[i] << 10), 6);
        if (dAllocR[i] > 0) put_bits((uint16_t)(sfR[i] << 10), 6);
    }

    for (int j = 0; j < NBLK; j++)
        for (int i = 0; i < NSUB; i++) {
            if (dAllocL[i] > 0) put_bits((uint16_t)codeL[j * NSUB + i], dAllocL[i]);
            if (dAllocR[i] > 0) put_bits((uint16_t)codeR[j * NSUB + i], dAllocR[i]);
        }

    if (bitpos + 8 <= cfg.frame_bytes * 8)
        put_bits(0, 8);

    /* flush the pending partial byte(s); low bits are already zero */
    {
        int k = (nb + 7) / 8;
        for (int j = 0; j < k; j++)
            *bout++ = (uint8_t)(bacc >> (56 - 8 * j));
    }
    return cfg.frame_bytes;
}

/* ------------------------------------------------------------------ */
/* WAV reading — identical structure/semantics to wav2pasc.c            */
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
    int      bits;
    int      sbytes;
    uint32_t datalen;
    uint32_t consumed;
} wav;

static uint8_t frame_raw[FRAME_SAM * 2 * 3];
static double pcmL[FRAME_SAM], pcmR[FRAME_SAM];

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
            if (fmt_tag == 0xFFFE) {
                if (r < 40 || rd16(fmt + 16) < 22) {
                    fprintf(stderr, "%s: bad extensible fmt chunk\n", name);
                    return -1;
                }
                fmt_tag = rd16(fmt + 24);
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

static int read_pcm_frame(void)
{
    size_t want = FRAME_SAM * 2 * (size_t)wav.sbytes;
    if (wav.datalen != 0xFFFFFFFFu && wav.datalen - wav.consumed < want)
        want = wav.datalen - wav.consumed;
    size_t got = fread(frame_raw, 1, want, wav.f);
    wav.consumed += (uint32_t)got;
    uint32_t nfr = (uint32_t)(got / (2u * (size_t)wav.sbytes));
    if (wav.bits == 16)
        K->deint16((const int16_t *)frame_raw, pcmL, pcmR, nfr);
    else
        K->deint24(frame_raw, pcmL, pcmR, nfr);
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

#ifdef FK_X86
    int asm_level = pick_asm_level();
    if (asm_level >= ASM_LEVEL_AVX512 && avx512_available())
        K = &kf_ops_avx512;
    else if (asm_level >= ASM_LEVEL_AVX2 && avx2_available())
        K = &kf_ops_avx2;
    fprintf(stderr, "wav2pasc: using %s kernels\n",
            asm_level_name(asm_level));
#endif

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
        uint8_t pre[2] = { cfg.mpp_rate_id, 0x00 };
        fwrite(pre, 1, 2, out);
    }

    init_tables();

    int known_len = wav.datalen != 0xFFFFFFFFu;
    uint32_t nframes = known_len
        ? wav.datalen / (2u * (uint32_t)wav.sbytes) / FRAME_SAM : 0;

    unsigned ndcc_i = 1;

    for (uint32_t fr = 0; ; fr++) {
        if (read_pcm_frame() < FRAME_SAM) break;
        K->analysis(pcmL, pcmR, xbufL, xbufR, coeff, cosT, SL, SR);

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
                if ((fr + 1) % 49 == 0) ndcc_i++;
            }
        } else {
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
