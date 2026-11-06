cmake_minimum_required(VERSION 3.10)
project(wav2pasc_fast C)

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

add_executable(wav2pasc_fast
  wav2pasc_fast.c
  fast_kernels_scalar.c
  fast_kernels_avx2.c
  fast_kernels.h
  ../wav2pasc/pasc_coeffs.h
)

target_include_directories(wav2pasc_fast PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../wav2pasc
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

if(NOT MSVC)
  target_link_libraries(wav2pasc_fast PRIVATE m)
endif()

if(MSVC)
  target_compile_definitions(wav2pasc_fast PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

install(TARGETS wav2pasc_fast RUNTIME DESTINATION bin)
