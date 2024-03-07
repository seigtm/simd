# SIMD SPbPU laboratory assignments

## Introduction

This repository provides sample C++ code illustrating the utilization of MMX, SSE, SSE2, and AVX
instructions.

## Code Overview

The repository consists of two source files:

1. `sample_code.cpp`:
   - Demonstrates the usage of **MMX, SSE, and SSE2 instructions**.
   - Performs operations such as comparing vectors, adding elements, computing square roots, and
     finding minimum elements.
   - Utilizes both inline assembly and intrinsic functions.
2. `mmx_sse_avx.cpp`:
   - Expands the usage to include AVX instructions along with MMX, SSE, and SSE2.
   - Implements functions using data formats `__m64`, `__m128`, and `__m256`.
   - Performs vector addition, square root calculation, and comparison of elements to find the
     minimum.

## Instructions

```bash
clang++ ./sample_code.cpp -std=c++20 -Ofast -march=native -o sample_code.exe
clang++ ./mmx_sse_avx.cpp -std=c++20 -Ofast -march=native -o mmx_sse_avx.exe
```

## Additional Information

- **Course**: Peter the Great St. Petersburg Polytechnic University (SPbPU), Computer Architecture.
- **Teachers**: Molodyakov S.A., Militsyn A.V.
