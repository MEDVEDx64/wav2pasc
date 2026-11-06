cmake_minimum_required(VERSION 3.10)
project(wav2pasc C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# Encoder: SIMD-accelerated (AVX-512F / AVX2 / scalar) PASC/MP1 encoder.
add_executable(wav2pasc
  wav2pasc.c
  fast_kernels_scalar.c
  fast_kernels_avx2.c
  fast_kernels_avx512.c
  fast_kernels.h
  pasc_coeffs.h
)

target_include_directories(wav2pasc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Decoder: plain C, no SIMD.
add_executable(pasc2wav
  pasc2wav.c
  pasc_coeffs.h
)

target_include_directories(pasc2wav PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# The AVX2 translation unit needs the AVX2 instruction set plus FP
# contraction disabled (no FMA fusion => bit-identical mul/add pairs).
# MSVC never fuses mul+add under the default /fp:precise.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(fast_kernels_avx2.c PROPERTIES
    COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
elseif(MSVC)
  set_source_files_properties(fast_kernels_avx2.c PROPERTIES
    COMPILE_OPTIONS "/arch:AVX2")
endif()

# The AVX-512 translation unit needs AVX-512F (and only AVX-512F;
# the 256-bit integer ops on cvtpd2dq results use VEX-encoded AVX2
# forms, available whenever AVX-512F is).  FP contraction is disabled
# for the same bit-exactness reasons as the AVX2 TU.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(fast_kernels_avx512.c PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-ffp-contract=off")
elseif(MSVC)
  set_source_files_properties(fast_kernels_avx512.c PROPERTIES
    COMPILE_OPTIONS "/arch:AVX512")
endif()

if(NOT MSVC)
  target_link_libraries(wav2pasc PRIVATE m)
  target_link_libraries(pasc2wav PRIVATE m)
endif()

if(MSVC)
  target_compile_definitions(wav2pasc PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(pasc2wav PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

install(TARGETS wav2pasc pasc2wav RUNTIME DESTINATION bin)
