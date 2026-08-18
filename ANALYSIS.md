# DCC2WAV.EXE PASC Encoder — Reverse Engineering Notes

Win16 Borland Pascal app (NE, 16 segments). Encoder logic lives in segments:
1 (convertor loop), 2 (TMPEGFrame/TPascFrameEncoder/TAllocator/frame packer),
3 (TPascFilter: polyphase analysis+synthesis), 4 (TRK/MPP stream I/O),
15 (RTL math helpers: :0x4AF=fxch+fdivp, :0x4B2=fdivp, :0x719=LN, :0x71D=EXP,
:0x711=COS, :0xDCF=ShlLong, :0xD06=LongDiv, :0xDAC=ShrLong).

Object model (recovered):
- TPascFrameEncoder subclasses TMPEGFrame; TAllocator (created by encoder) holds
  the psychoacoustic tables; TPascFilter is mode 0 = synthesis / 1 = analysis.
- Internal far calls: operand selector is a fixup-chain pointer, the OFFSET is
  the real target; `lcall 0xffff:off` calls are patched cross-segment calls.

## Encoder pipeline (per frame = 384 samples/channel)

1. **Input** (seg03:0x2CF1): WAV 44100 (no prefilter) / 22050 / 11025 (17-tap
   FIR prefilters = interpolation filters, two coefficient sets). PCM int32 =
   s16 << 2, interleaved.
2. **Polyphase analysis** (seg03:0xAA4), 12 blocks/frame:
   - window W[512]: shift up by 32 (newest sample at index 0: block sample i -> W[31-i]),
     x = int32/2^17
   - Z[k] = sum_{j=0..7} W[64j+k] * (COF[64j+k]/2^21)
   - S[block*32+m] = sum_{k=0..63} cos((k-16)(2m+1)pi/64) * Z[k]
   - peak trackers (int32 domain, reset each frame, used for the header word)
3. **Tables** (TMPEGFrame ctor seg02:0xA4, n=2..15):
   offset[n]=2^-n, mult2[n]=2^(1-n), mult[n]=2^n/(2^n-1),
   maskA[n]=2^(n-1), maskB[n]=0xFFFFFFFF<<n (sign-extension mask),
   scaleTable[i]=2^(1-i/3), i=0..62
4. **Psychoacoustics** (TAllocator):
   - M[i] = 10^(min(T(f_lo),T(f_hi))/10), f_lo = 20 (i=0) else 700*i,
     f_hi = 700*(i+1);  T(x)= 3.64 x^-0.8 - 6.5 e^(-0.6(x-3.3)^2) + 0.001 x^4 + 5,
     x=f/1000, clamped [x>=0.02, T<=60], special T(3.3)=0.019115056
   - spreading SP[i][j]: j<=i-2: 0.1*(f_j/f_i)^10 (upward); j>=i:
     0.1*(f_i/f_j)^27.8 (downward); j=i-1 = 0. Band edges from 689.0625 Hz grid
     (f_j = 20 Hz for j=0)
   - K1 = 2^-35/12, K2 = 2^-34/12
   - energy[i] = sum_j S[j][i]^2 / 12 (L only! R energies are NOT divided —
     original quirk), max[i] = max |S|
   - thr[i] = K1*M[i] + sum_j SP[i][j]*energy[j]
   - **SMR[i] = ln(max[i]^2/(3*thr[i])) / (8 ln 8)**  (= log_8(ratio)/8)
     (computed via LN + fxch-fdivp; the division order matters!)
     MAXSNR sentinel = -10000.0
5. **Allocation** (seg02:0x105E) — CORRECTED & VERIFIED (byte-exact):
   - DGROUP working allocs zeroed; TAllocator allocL/allocR persist across frames
   - pass4: energy>=thr -> alloc=5; elif alloc>0: alloc--
   - pass5: allocL=allocR=MAX (jae: R=L else L=R)  [NOT min!]
   - pass6: alloc==5 -> alloc2=5; energy>=K2 -> alloc2--; alloc==0 -> alloc2=5
   - waterfill: snr field = log4(max^2/(3*thr)) (ln(ratio)/(2 ln 2));
     initial grants: a1>0 -> dg=2, snr-=2.0, budget-=30
     loop: pick max-snr band (first on ties); step=2,cost=30 if dg==0 else 1,12;
     invalidate if cost>budget or dg==15 or (a2>0 and a1==0); else dg+=step,
     snr-=step, budget-=cost; exit when best==-10000 (all MAXSNR)
   - budget = 3040/2 = 1520 per channel:
     for each band with alloc>0: dg=2, snr-=8, budget-=30
     loop: pick max snr; stop if == -10000;
       step = dg==0?2:1, cost = dg==0?30:12
       invalidate (snr=-10000) if cost>budget or dg==15
       (gates alloc2<=0 / alloc!=0 only skip the invalidation)
       if snr != -10000: dg+=step, snr-=step, budget-=cost
6. **Scale factors** (seg02:0x6B7): sf=62; while max >= scaleTable[sf] && sf>0: sf--
7. **Quantizer** (seg02:0x832): n=alloc; qs = S/scaleTable[sf];
   clamp qs >= 1.0 -> 0.9999998807907104;
   q = (1-2^-n)*qs - 2^-n; v = round(q*2^23);
   code16 = ((v<<8) ^ 0x80000000) >> 16   (offset binary top bits)
8. **Frame packer** (seg02:0x2EBE + 0x3447/0x3597/0x3607):
   - 572-byte buffer, output 416 (0x1A0) bytes (+2 peak = 418-byte PASC frame)
   - byte[i] = ((allocL-1)<<4) | (allocR-1), i=0..31 (0 if alloc==0)
   - bitpos = 0x100; MSB-first bit writer (fixed mask table 1<<(16-p) =>
     writes the TOP count bits of a 16-bit word; scale factors must be
     pre-shifted, the original multiplies an accumulator by 4 per band)
   - for i: if allocL>0: 6 bits sfL; then all sfR
   - for j=0..11, i=0..31: if allocL>0: allocL bits of codeL;
     if allocR>0: allocR bits of codeR  (L/R interleaved per band!)
   - trailing marker byte; frame truncated to 416
   - peak header word = (peakR>>9)<<8 | (peakL>>9), peaks clamped to 0x1FFFF,
     written as 2 LE bytes before the frame
9. DCCFIR.COF: 512 LE int32 = polyphase prototype; /2^21 for analysis,
   /65536 for synthesis.

## Decoder (for verification)
- allocations: nibble+1 if >0 (seg02:0x2F75); 16-bit-window MSB-first reader
  (seg02:0x2FEE); silent bands get sf=0x3F
- dequant (seg02:0x2D8): v = code ^ 2^(n-1); if (v & 2^(n-1)): v |= 0xFFFFFFFF<<n
  (sign extension); S = (v/2^(n-1) + 2^(1-n)) * 2^n/(2^n-1) * scaleTable[sf]
- synthesis (seg03:0x112B): 1024-window, shift 64; W[k] += V[k][m]*S[m] with
  V[k][m] = cos((k+16)(2m+1)pi/64) for ALL k=0..63 (FilterType 0 builds the
  full 64x32 matrix; FilterType 1 builds only the 32x64 analysis matrix —
  same memory, mode-dependent); temp[i] = W[OT[i]]*coeffA[i],
  OT[i] = 64*(i/32) + (i%64); out[m] = sum_{k=0..15} temp[k*32+m] * 32768,
  clamped to int16; written as int32 (=s16, no <<2)

## Verification (FINAL)
- wav2pasc output is BYTE-FOR-BYTE IDENTICAL to DCC2WAV.EXE (Adaptive mode)
  on test/sample.wav: 26,487/26,487 records, 11,124,542/11,124,542 bytes.
- Method that resolved the last unknowns: unicorn emulation of the original
  seg02 code (fn 0xCB5/0xDE9/0xEF4/0x105E) with FP-helper far-calls patched
  to inline x87 stubs (LN=fyl2x, EXP=f2xm1 sequence, fdivp, fxch+fdivp).
  Pitfalls: SS segment must not overlap the object area; near-call stubs.
- Analysis<->synthesis round-trip: corr 1.000000, unity gain (latency 962).
- Full encode/decode round-trip: ~26-28 dB SNR at 384 kbps.

## .mpp container (from sample_original_encoder.mpp)
- 2-byte preamble: 2C 00
- records: [sync FF FF C0 00][416-byte PASC frame] = 420 bytes
- record count = floor(wav_stereo_frames / 384) (trailing partial dropped)

## Loose ends / caveats
- x87 80-bit intermediates vs double: not bit-exact, algorithmically identical
- "Standard conversion" fixed-table mode reads DGROUP:0x40 which holds RIFF
  scratch in the file image — appears to be dead code; only the psychoacoustic
  path (encoder mode 1) is implemented
- exact RTL Round = frndint (nearest-even); we use nearbyint()


## Extensions (post-verification work)

- 24-bit input: normalize PCM to [-1,1) doubles before the analysis
  filter; identical bitstream for 16-bit input (verified byte-exact).
- 48 kHz: bw = 1500 Hz for the psychoacoustic tables; frame payload
  380 bytes; per-channel budget 1392 bits ((380*8 - 32 - 256)/2);
  MPEG header byte2 = 0xC4; .mpp records 384 bytes, preamble 30 00.
- MP1 output (-f 1): DCC/newdccu padding pattern — counter i starts at 1,
  frames with even i padded (420 B), extra increment after every 49th
  written frame => 24 padded + 25 unpadded per group, average exactly
  384 kbps. Byte-exact vs newdccu's mpp->mp1 conversion.
- Build: CMake (MSVC/GCC/Clang).

Measured 24-bit benefit: +21.5 dB SNR on -60 dBFS content (54.2 vs 32.7 dB
vs 24-bit master); no measurable change on loud material (the codec's own
quantization noise dominates there).
