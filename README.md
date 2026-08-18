*This project is AI forged: no way on Earth I could make it
without the might of GLM-5.2.*

# wav2pasc / pasc2wav

The PASC (Precision Adaptive Sub-band Coding) encoder and decoder used by
Digital Compact Cassette, extracted from the Win16 application
**DCC2WAV.EXE** ("DCC to Wave Converter", 1996) by reverse engineering its
16-bit Borland Pascal code (see `ANALYSIS.md` for the complete
disassembly notes).

PASC is an MPEG-1 Audio Layer 1 derivative: a 512-tap polyphase filter bank
with 32 subbands, psychoacoustic bit allocation and 6-bit scale factors,
running at 384 kbit/s stereo in 416-byte frames (44.1 kHz).

Output is **byte-for-byte identical** to the original application
(verified over a 230 s, 26,487-frame test file) and to the `newdccu`
reference conversion for MP1 output.

## Extensions over DCC2WAV.EXE

- **24-bit input WAVs** — the original only accepted 8/16-bit input.
  Layer 1's per-subband quantizers reach 15 bits *relative to each band's
  level*, i.e. well below the 16-bit input noise floor for quiet bands.
  On quiet material (fades, reverb tails, low-level tones) this recovers
  up to ~20 dB of fidelity that the original application threw away:
  measured 54.2 dB vs 32.7 dB SNR on a -60 dBFS test signal.
- **48 kHz sample rate** — MPEG Layer 1 supports it natively (380-byte
  frames, `0xC4` header); PASC/DCC never did. 48 kHz MP1 output is
  standards-compliant and plays everywhere.
- **Direct MP1 output** — MPEG-1 Layer I elementary streams playable by
  any MPEG audio decoder.

## Build

    cmake -B build && cmake --build build

(Windows/MSVC and GCC/Clang on Unix are supported by the CMake setup.)

## Usage

    wav2pasc [options] input.wav output.mpp

    -f, --format 0|1   output format (default 0)
                         0 = PASC .mpp container (DCC-Studio compatible)
                         1 = MPEG-1 Layer I .mp1 elementary stream

    pasc2wav input.[mpp|mp1] output.wav [--raw] [--24]
      --raw  input is bare 416-byte PASC frames (44.1 kHz)
      --24   write 24-bit WAV output (default 16-bit)

Input: 44100 or 48000 Hz stereo PCM WAV, 16 or 24-bit (WAVE_FORMAT_EXTENSIBLE
and streaming WAVs with unknown data size handled). For other rates/channels
resample first, e.g. `sox in.wav -r 44100 -c 2 -b 24 out.wav`.

Both tools support piped I/O: use `-` as a file name for stdin/stdout,
e.g. `ffmpeg -i track.flac -f wav - | wav2pasc - out.mpp` or
`wav2pasc in.wav - -f 1 | pasc2wav - play.wav`. WAV written to a pipe uses
unknown chunk sizes (0xFFFFFFFF), understood by ffmpeg/sox.

Note: a 48 kHz `.mpp` is emitted for completeness, but DCC-Studio/decks
expect 44.1 kHz PASC; use `-f 1` for 48 kHz content.

## File formats

`.mpp` PASC container (byte-exact with DCC2WAV.EXE):

    [2-byte preamble: rate ID (0x2C=44100, 0x30=48000), 0x00]
    [420-byte record] * N        (44.1 kHz)
    [384-byte record] * N        (48 kHz)
     record: sync FF FF C0/C4 00 + 416/380-byte PASC frame
     frame: 32 allocation bytes (nibbles = alloc-1 per subband/channel),
            then MSB-first bit stream: per band 6-bit scale factors
            interleaved L,R; then per block/band the variable-length
            sample codes, L then R per band.
    N = floor(wav_stereo_frames / 384)  (trailing partial frame dropped)

`.mp1` MPEG-1 Layer I elementary stream (byte-exact with `newdccu`):

    [FF FF Cx 00 header][payload] — 416/420-byte frames at 44.1 kHz with
    the DCC padding pattern (24 of every 49 frames padded → average
    exactly 384 kbps), 384-byte frames at 48 kHz.

## Pipeline (faithful to the original)

1. PCM normalized to [-1, 1); for 16-bit input this is identical to the
   original's int32 = s16<<2 then /2^17 path. The 17-tap FIR pre-filters
   of the original (11.025/22.05 kHz upsample only) are not implemented —
   resample externally instead.
2. 512-tap polyphase analysis, prototype window = DCCFIR.COF / 2^21,
   cosine matrix M[m][k] = cos((k-16)(2m+1)pi/64), 12 blocks of 32 samples
   per frame
3. Psychoacoustics: Terhardt absolute threshold
   T(f) = 3.64 x^-0.8 - 6.5 e^(-0.6(x-3.3)^2) + 0.001 x^4 + 5 dB (x=f/1kHz),
   upward spreading 0.1 (f_j/f_i)^10 (includes the c=r-1 diagonal — original
   quirk), downward spreading 0.1 (f_i/f_j)^27.8,
   threshold floor 2^-35/12 * 10^(T/10)
4. Thresholds thr[i] = K1*M[i] + sum_j SP[i][j]*energy[j] with the exact
   original quirks: L energies divided by 12, R energies NOT (sic), and
   stereo max-coupling of the activity state. SMR = log4(max^2/(3*thr)),
   gated water-filling with a budget of 1520 bits/channel at 44.1 kHz
   (1392 at 48 kHz); activation costs 30 = 2*12 + 6 scale-factor bits,
   each further bit costs 12
5. Scale factors from the 2^(1-i/3) table (2 dB steps, MPEG Layer I grid)
6. Quantization q = (1-2^-n) clip(S/scale) - 2^-n, offset-binary codes
7. MSB-first bit packing

## Verification

- `-f 0`, 16-bit, 44.1 kHz: **byte-for-byte identical** to DCC2WAV.EXE
  (Adaptive mode) on a 230 s test file (26,487 records, 11,124,542 bytes).
- `-f 1`: **byte-for-byte identical** to converting that .mpp with the
  `newdccu` reference tool (DCCU padding pattern).
- ffprobe reports exactly 384 kbps for both rates; ffmpeg decodes all
  outputs with correct durations.
- 24-bit benefit: 54.2 dB vs 32.7 dB SNR against the 24-bit master on a
  -60 dBFS test tone (+21.5 dB).
- 48 kHz: encodes, decodes (own decoder + ffmpeg), 384-byte frames.
- The psychoacoustic tables were cross-validated by emulating the original
  16-bit code (unicorn) directly during reverse engineering.

The codec latency is 962 samples (encode 481 + decode 481).