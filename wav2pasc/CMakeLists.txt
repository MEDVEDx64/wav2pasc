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

add_executable(wav2pasc wav2pasc.c pasc_coeffs.h)
add_executable(pasc2wav pasc2wav.c pasc_coeffs.h)

if(NOT MSVC)
  target_link_libraries(wav2pasc PRIVATE m)
  target_link_libraries(pasc2wav PRIVATE m)
endif()

if(MSVC)
  target_compile_definitions(wav2pasc PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(pasc2wav PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

install(TARGETS wav2pasc pasc2wav RUNTIME DESTINATION bin)