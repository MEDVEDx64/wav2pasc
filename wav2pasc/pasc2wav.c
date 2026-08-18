/*
 * pasc2wav — PASC / MPEG-1 Layer I decoder
 * (from DCC2WAV.EXE seg02:0x2D8/0x2F75 + seg03:0x112B)
 * Round-trip companion to wav2pasc; also useful on its own.
 *
 * Input containers (autodetected):
 *   .mpp: [2-byte rate ID (0x2C=44100, 0x30=48000)][fixed-size records]
 *         44.1 kHz: 420-byte records (FF FF C0 00 + 416-byte frame)
 *         48   kHz: 384-byte records (FF FF C4 00 + 380-byte frame)
 *   .mp1: bare MPEG-1 Layer I elementary stream; frame size from the
 *         header (44.1 kHz: 416/420 via padding bit, 48 kHz: 384)
 *   --raw: bare 416-byte PASC frames (44.1 kHz only)
 *
 * Output: 16-bit stereo WAV (default) or 24-bit with --24.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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

#define NSUB 32
#define NBLK 12

static double coeffA[512];      /* COF/65536 (synthesis) */
static double V[64][NSUB];      /* synthesis matrix */
static double winL[1024], winR[1024];
static double scaleT[64];
static double mult2T[16], multT[16];

static double SL[NBLK*NSUB], SR[NBLK*NSUB];

static int out_bits = 16;

static void init(void)
{
    for (int i = 0; i < 512; i++)
        coeffA[i] = (double)pasc_cof[i] / 65536.0;
    /* synthesis matrix: FilterType 0 builds ALL 64 rows as cos((k+16)(2m+1)pi/64)
       (seg03:0x31D: type 0 -> first loop fills rows 0..63, cols 0..31) */
    for (int k = 0; k < 64; k++)
        for (int m = 0; m < NSUB; m++)
            V[k][m] = cos((k + 16) * (2*m + 1) * M_PI / 64.0);
    for (int i = 0; i < 64; i++)
        scaleT[i] = pow(2.0, 1.0 - (double)i / 3.0);
    for (int n = 2; n <= 15; n++) {
        double p2n = (double)(1 << n);
        mult2T[n] = 2.0 / p2n;
        multT[n] = 1.0 + 1.0 / (p2n - 1.0);
    }
}

/* bit reader (mirror of the packer) */
static const uint8_t *fbuf;
static int bitpos;
static unsigned get_bits(int n)
{
    unsigned v = 0;
    for (int b = 0; b < n; b++) {
        v <<= 1;
        if (fbuf[bitpos >> 3] & (0x80 >> (bitpos & 7))) v |= 1;
        bitpos++;
    }
    return v;
}

/* dequantize one frame's subband samples (seg02:0x2D8) */
static void dequantize(const uint8_t *allocL, const uint8_t *allocR,
                       const uint8_t *sfL, const uint8_t *sfR,
                       const uint16_t *cL, const uint16_t *cR)
{
    for (int j = 0; j < NBLK; j++)
        for (int i = 0; i < NSUB; i++) {
            int idx = j*NSUB + i;
            if (allocL[i] > 0) {
                int n = allocL[i];
                uint32_t v = cL[idx];
                uint32_t maskA = 1u << (n-1);
                uint32_t maskB = 0xFFFFFFFFu << n;
                v ^= maskA;
                if (v & maskA)          /* c MSB was 0 -> negative: sign-extend */
                    v |= maskB;
                SL[idx] = (((double)(int32_t)v / (double)maskA) + mult2T[n])
                          * multT[n] * scaleT[sfL[i]];
            } else SL[idx] = 0.0;
            if (allocR[i] > 0) {
                int n = allocR[i];
                uint32_t v = cR[idx];
                uint32_t maskA = 1u << (n-1);
                uint32_t maskB = 0xFFFFFFFFu << n;
                v ^= maskA;
                if (v & maskA)
                    v |= maskB;
                SR[idx] = (((double)(int32_t)v / (double)maskA) + mult2T[n])
                          * multT[n] * scaleT[sfR[i]];
            } else SR[idx] = 0.0;
        }
}

/* synthesis one block (seg03:0x112B): 32 stereo samples out */
static void synth_block(int base, double *out)
{
    for (int i = 1023; i >= 64; i--) {
        winL[i] = winL[i-64];
        winR[i] = winR[i-64];
    }
    for (int i = 0; i < 64; i++) { winL[i] = 0.0; winR[i] = 0.0; }

    for (int k = 63; k >= 0; k--)
        for (int m = 0; m < NSUB; m++) {
            winL[k] += V[k][m] * SL[base*NSUB + m];
            winR[k] += V[k][m] * SR[base*NSUB + m];
        }

    double t1[512], t2[512];
    for (int i = 0; i < 512; i++) {
        int ot = 64 * (i / 32) + (i % 64);
        t1[i] = winL[ot] * coeffA[i];
        t2[i] = winR[ot] * coeffA[i];
    }
    for (int m = 0; m < NSUB; m++) {
        double al = 0.0, ar = 0.0;
        for (int k = 0; k < 16; k++) {
            al += t1[k*32 + m];
            ar += t2[k*32 + m];
        }
        out[(base*NSUB + m)*2 + 0] = al;
        out[(base*NSUB + m)*2 + 1] = ar;
    }
}

/* ------------------------------------------------------------------ */

static void make_wav_header(uint8_t hdr[44], uint32_t rate, int bits, uint32_t data_bytes)
{
    int sbytes = bits / 8;
    uint32_t byterate = rate * 2 * sbytes;
    uint32_t riff = (data_bytes == 0xFFFFFFFFu) ? 0xFFFFFFFFu : 36 + data_bytes;
    memcpy(hdr, "RIFF", 4);
    hdr[4]=riff&0xFF; hdr[5]=(riff>>8)&0xFF; hdr[6]=(riff>>16)&0xFF; hdr[7]=(riff>>24)&0xFF;
    memcpy(hdr+8, "WAVE", 4);
    memcpy(hdr+12, "fmt ", 4);
    hdr[16]=16; hdr[17]=0; hdr[18]=0; hdr[19]=0;
    hdr[20]=1; hdr[21]=0;                     /* PCM */
    hdr[22]=2; hdr[23]=0;                     /* stereo */
    hdr[24]=rate&0xFF; hdr[25]=(rate>>8)&0xFF; hdr[26]=(rate>>16)&0xFF; hdr[27]=(rate>>24)&0xFF;
    hdr[28]=byterate&0xFF; hdr[29]=(byterate>>8)&0xFF; hdr[30]=(byterate>>16)&0xFF; hdr[31]=(byterate>>24)&0xFF;
    hdr[32]=(uint8_t)(2*sbytes); hdr[33]=0;   /* block align */
    hdr[34]=(uint8_t)bits; hdr[35]=0;         /* bits */
    memcpy(hdr+36, "data", 4);
    hdr[40]=data_bytes&0xFF; hdr[41]=(data_bytes>>8)&0xFF; hdr[42]=(data_bytes>>16)&0xFF; hdr[43]=(data_bytes>>24)&0xFF;
}

static void usage(const char *prog)
{
    printf("usage: %s input.[mpp|mp1] output.wav [--raw] [--24]\n"
           "\n"
           "options:\n"
           "  --raw   input is bare 416-byte PASC frames (44.1 kHz)\n"
           "  --24    write 24-bit WAV output (default 16-bit)\n"
           "\n"
           "containers (autodetected): .mpp (2-byte rate-ID preamble + fixed\n"
           "records) or .mp1 (bare MPEG-1 Layer I stream, variable frames)\n"
           "use '-' as the file name to read from stdin / write to stdout\n"
           "(streamed WAV output uses unknown sizes, 0xFFFFFFFF)\n",
           prog);
}

int main(int argc, char **argv)
{
    const char *inpath = NULL, *outpath = NULL;
    int raw = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--raw")) { raw = 1; continue; }
        if (!strcmp(argv[i], "--24")) { out_bits = 24; continue; }
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        if (!inpath) inpath = argv[i];
        else if (!outpath) outpath = argv[i];
        else { fprintf(stderr, "too many arguments\n"); return 1; }
    }
    if (!inpath || !outpath) { usage(argv[0]); return 1; }

    FILE *in = stdin;
    if (strcmp(inpath, "-")) {
        in = fopen(inpath, "rb");
        if (!in) { perror(inpath); return 1; }
    } else {
        setbin_stdin();
    }
    FILE *out = stdout;
    if (strcmp(outpath, "-")) {
        out = fopen(outpath, "wb");
        if (!out) { perror(outpath); return 1; }
    } else {
        setbin_stdout();
    }

    /* detect container (sequential-only; peek bytes are reused below) */
    int is_mpp = 0, mpp_rec = 420, mp1_rate = 44100;
    uint8_t peek[2];
    int have_peek = 0;
    if (!raw) {
        if (fread(peek, 1, 2, in) != 2) { fprintf(stderr, "short file\n"); return 1; }
        if ((peek[0] == 44 || peek[0] == 48) && peek[1] == 0x00) {
            is_mpp = 1;
            if (peek[0] == 48) mpp_rec = 384;
        } else {
            /* bare MP1: the peek bytes are the first two header bytes */
            if (peek[0] != 0xFF || (peek[1] & 0xF0) != 0xF0) {
                fprintf(stderr, "not an .mpp or .mp1 stream\n");
                return 1;
            }
            have_peek = 1;
        }
    }

    int out_seekable = (fseek(out, 0, SEEK_CUR) == 0);

    init();
    uint8_t rec[424];
    uint32_t frames = 0;
    double pcm[NBLK*NSUB*2];
    uint64_t total_pcm_frames = 0;
    int hdr_written = 0;
    uint32_t rate = raw ? 44100u : (is_mpp ? (uint32_t)(mpp_rec == 384 ? 48000 : 44100) : 0u);

    for (;;) {
        const uint8_t *fr;

        if (raw) {
            if (fread(rec, 1, 416, in) != 416) break;
            fr = rec;
        } else if (is_mpp) {
            if (fread(rec, 1, (size_t)mpp_rec, in) != (size_t)mpp_rec) break;
            if (rec[0] != 0xFF || rec[1] != 0xFF || (rec[2] & 0xF0) != 0xC0) {
                fprintf(stderr, "warning: bad sync at record %u\n", frames);
                break;
            }
            fr = rec + 4;
        } else {
            /* bare MP1: [4-byte header][payload] */
            uint8_t hdr[4];
            if (have_peek) {          /* first frame: 2 bytes already read */
                hdr[0] = peek[0]; hdr[1] = peek[1];
                if (fread(hdr + 2, 1, 2, in) != 2) break;
                have_peek = 0;
            } else {
                if (fread(hdr, 1, 4, in) != 4) break;
            }
            if (hdr[0] != 0xFF || (hdr[1] & 0xF0) != 0xF0 || (hdr[1] & 0x06) != 0x06) {
                fprintf(stderr, "warning: bad MPEG header at frame %u\n", frames);
                break;
            }
            if ((hdr[2] & 0xF0) != 0xC0) {
                fprintf(stderr, "warning: not 384 kbps at frame %u\n", frames);
                break;
            }
            int sr = (hdr[2] >> 2) & 3;
            int pad = (hdr[2] >> 1) & 1;
            int fsz;
            switch (sr) {
            case 0: fsz = 416 + 4*pad; break;   /* 44.1 kHz */
            case 1: fsz = 384; break;           /* 48 kHz   */
            case 2: fsz = 576; break;           /* 32 kHz   */
            default:
                fprintf(stderr, "bad sample rate at frame %u\n", frames);
                return 1;
            }
            if (fsz > 424) {
                fprintf(stderr, "frame too large (%d)\n", fsz);
                return 1;
            }
            mp1_rate = (sr == 1) ? 48000 : (sr == 2 ? 32000 : 44100);
            rate = mp1_rate;
            if (fread(rec, 1, (size_t)fsz - 4, in) != (size_t)fsz - 4) break;
            fr = rec;                              /* payload (header consumed above) */
        }

        /* write the WAV header once the rate is known (first record);
           non-seekable outputs get unknown (0xFFFFFFFF) sizes */
        if (!hdr_written) {
            uint8_t hdr[44];
            make_wav_header(hdr, rate, out_bits, 0xFFFFFFFFu);
            fwrite(hdr, 1, 44, out);
            hdr_written = 1;
        }

        uint8_t allocL[NSUB], allocR[NSUB], sfL[NSUB], sfR[NSUB];
        uint16_t cL[NBLK*NSUB], cR[NBLK*NSUB];

        for (int i = 0; i < NSUB; i++) {
            uint8_t b = fr[i];
            allocL[i] = b >> 4;
            if (allocL[i] > 0) allocL[i]++;
            allocR[i] = b & 0xF;
            if (allocR[i] > 0) allocR[i]++;
        }
        fbuf = fr; bitpos = 0x100;
        for (int i = 0; i < NSUB; i++) {
            sfL[i] = allocL[i] > 0 ? (uint8_t)get_bits(6) : 0;
            sfR[i] = allocR[i] > 0 ? (uint8_t)get_bits(6) : 0;
        }
        for (int j = 0; j < NBLK; j++)
            for (int i = 0; i < NSUB; i++) {
                cL[j*NSUB+i] = allocL[i] > 0 ? (uint16_t)get_bits(allocL[i]) : 0;
                cR[j*NSUB+i] = allocR[i] > 0 ? (uint16_t)get_bits(allocR[i]) : 0;
            }

        dequantize(allocL, allocR, sfL, sfR, cL, cR);
        for (int b = 0; b < NBLK; b++)
            synth_block(b, pcm);

        for (int i = 0; i < NBLK*NSUB; i++) {
            if (out_bits == 24) {
                long l = lround(pcm[i*2]   * 8388608.0);
                long r = lround(pcm[i*2+1] * 8388608.0);
                if (l >  8388607) l =  8388607;
                if (l < -8388608) l = -8388608;
                if (r >  8388607) r =  8388607;
                if (r < -8388608) r = -8388608;
                uint8_t sb[6] = {
                    (uint8_t)l, (uint8_t)(l>>8), (uint8_t)(l>>16),
                    (uint8_t)r, (uint8_t)(r>>8), (uint8_t)(r>>16) };
                fwrite(sb, 1, 6, out);
            } else {
                long l = lround(pcm[i*2]   * 32768.0);
                long r = lround(pcm[i*2+1] * 32768.0);
                if (l >  32767) l =  32767;
                if (l < -32768) l = -32768;
                if (r >  32767) r =  32767;
                if (r < -32768) r = -32768;
                int16_t sl = (int16_t)l, sr = (int16_t)r;
                uint8_t sb[4] = {(uint8_t)sl, (uint8_t)(sl>>8), (uint8_t)sr, (uint8_t)(sr>>8)};
                fwrite(sb, 1, 4, out);
            }
        }
        total_pcm_frames += NBLK*NSUB;
        frames++;
    }

    /* finalize the WAV header: patch real sizes if the output is seekable */
    if (!hdr_written) {
        /* nothing decoded: emit an empty valid header anyway */
        if (!rate) rate = 44100;
        uint8_t hdr[44];
        make_wav_header(hdr, rate, out_bits, 0);
        fwrite(hdr, 1, 44, out);
    } else if (out_seekable) {
        int sbytes = out_bits / 8;
        uint32_t data_bytes = (uint32_t)(total_pcm_frames * 2 * sbytes);
        uint8_t hdr[44];
        make_wav_header(hdr, rate, out_bits, data_bytes);
        fseek(out, 0, SEEK_SET);
        fwrite(hdr, 1, 44, out);
    }

    if (in != stdin) fclose(in);
    if (out != stdout) fclose(out);
    else fflush(out);
    fprintf(stderr, "decoded %u frames (%llu samples, %u Hz, %d-bit%s)\n",
            frames, (unsigned long long)total_pcm_frames, rate, out_bits,
            out_seekable ? "" : ", streamed header");
    return 0;
}