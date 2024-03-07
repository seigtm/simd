/** @file mmx_sse_avx.cpp
 *  @brief ММХ, SSE, AVX commands test.
 *  @details `clang++ ./mmx_sse_avx.cpp -std=c++20 -Ofast -march=native`.
 *  @author Baranov Konstantin (seigtm) <gh@seig.ru>
 *  @version 1.0
 *  @date 2024-03-05
 */

#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>


namespace setm {

// Literal suffix for 8-bit unsigned integer.
static constexpr std::uint8_t operator"" _u8(unsigned long long v) { return static_cast<std::uint8_t>(v); }

// Literal suffix for 16-bit unsigned integer.
static constexpr std::uint16_t operator"" _u16(unsigned long long v) { return static_cast<std::uint16_t>(v); }

// Literal suffix for 32-bit unsigned integer.
static constexpr std::uint32_t operator"" _u32(unsigned long long v) { return static_cast<std::uint32_t>(v); }

// Prints an array with the corresponding message.
static const auto print_array(const auto arr, std::string_view message = "") {
    std::cout << message;
    for(const auto element : arr)
        std::cout << +element << ' ';
    std::cout << '\n';
}

// Prints a value in a binary representation.
static void print_binary(const auto value) {
    std::cout << std::bitset<std::numeric_limits<decltype(value)>::digits>{ value };
}

// Using MMX intrinsics, perform vector addition and store results.
const __m64 add_vectors(const std::array<std::uint16_t, 4>& a, const std::array<std::uint16_t, 4>& b) {
    const auto xmm0{ *reinterpret_cast<const __m64*>(a.data()) };
    const auto xmm1{ *reinterpret_cast<const __m64*>(b.data()) };

    return _mm_adds_pu16(xmm0, xmm1);
}

// Using SSE intrinsics, perform square root calculation and store result.
const __m128d sqrt_intrinsic(const std::array<double, 2>& a) {
    const auto xmm0{ *reinterpret_cast<const __m128d*>(a.data()) };

    return _mm_sqrt_pd(xmm0);
}

// Using AVX2 intrinsics, compare and store minimum elements
const __m256i compare_vectors_min(const std::array<std::uint8_t, 32>& a, const std::array<std::uint8_t, 32>& b) {
    const auto xmm0{ _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data())) };
    const auto xmm1{ _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data())) };

    return _mm256_min_epu8(xmm0, xmm1);
}

}  // namespace setm

int main() {
    // Checks if all instructions are found.
#if !(defined(__MMX__) &&  \
      defined(__SSE__) &&  \
      defined(__SSE2__) && \
      defined(__SSE3__) && \
      defined(__AVX__) &&  \
      defined(__AVX2__))

    std::cerr << "Some instructions are not found.\n";
    return EXIT_FAILURE;
#endif

    using namespace setm;

    // Task 1. Write a program to study MMX-SSE-AVX commands.
    // You should utilize at least two "special" commands along with a command for AVX registers.
    // The "special" commands include:
    //   - with saturation,
    //   - comparison,
    //   - permutations,
    //   - packing/unpacking,
    //   - SSE3, etc.

    // 1.1. Inline assembly to shuffle elements between xmm0 and xmm1 using shufps.
    {
        std::cout << "1.1. Use shufps to select elements from two registers:\n";
        alignas(sizeof(float) * 4) std::array xmm0{ 11.0f, 12.0f, 13.0f, 14.0f };
        alignas(sizeof(float) * 4) std::array xmm1{ 21.0f, 22.0f, 23.0f, 24.0f };
        constexpr auto bitmask{ 0b01'11'00'00_u32 };
        print_array(xmm0, "xmm0 = ");
        print_array(xmm1, "xmm1 = ");
        std::cout << "bitmask = ";
        print_binary(bitmask);
        std::cout << '\n';

        asm volatile(
            "movaps %0, %%xmm0\n\t"  // Load data to xmm0 register.
            "movaps %1, %%xmm1\n\t"  // Load data to xmm1 register.

            // The first two elements of destination register are overwritten with any two elements
            // of this register. The third and fourth element are overwritten with two elements
            // form the source register. The selection of elements is controlled by the pairs of
            // bits from the mask, interpreted as numbers from range 0-3.
            "shufps %2, %%xmm1, %%xmm0\n\t"  // Shuffle data.
            "movups %%xmm0, %0\n\t"          // Store data.

            : "+m"(xmm0)  // Input and output from/to xmm0 variable.
            // Input from xmm1 and bitmask as operands.
            // Bitmask is 0b00'11'00'00:
            // - 00 - element at index 0 from xmm1;
            // - 11 - element at index 3 from xmm1;
            // - 00 - element at index 0 from xmm0;
            // - 00 - element at index 0 from xmm0.
            : "m"(xmm1), "i"(bitmask));

        print_array(xmm0, "Shuffled xmm0 = ");
    }


    // 1.2. Inline assembly to compare elements in xmm0 and xmm1.
    {
        std::cout << "\n1.2. Compare elements in xmm0 and xmm1 by using pcmpeqb:\n";
        alignas(sizeof(std::uint32_t) * 4) std::array xmm0{ 1_u32, 2_u32, 3_u32, 4_u32 };
        alignas(sizeof(std::uint32_t) * 4) std::array xmm1{ 1_u32, 4_u32, 3_u32, 2_u32 };

        print_array(xmm0, "xmm0 = ");
        print_array(xmm1, "xmm1 = ");

        std::cout << "Compared elements: ";
        for(std::size_t index{}; index < xmm0.size(); ++index)
            std::cout << "(" << xmm0[index] << ", " << xmm1[index] << "); ";
        std::cout << '\n';

        asm volatile(
            "movaps %0, %%xmm0\n\t"  // Load data to xmm0 register.
            "movaps %1, %%xmm1\n\t"  // Load data to xmm1 register.

            "pcmpeqd %%xmm1, %%xmm0\n\t"  // Compare elements in xmm0 and xmm1.
            "movups %%xmm0, %0\n\t"       // Store data.

            : "+m"(xmm0)   // Input and output from/to xmm0 variable.
            : "m"(xmm1));  // Input from xmm1 as operand.

        for(const auto element : xmm0) {
            std::cout << "Binary representation: ";
            print_binary(element);
            std::cout << ", meaning that: " << ((0 == element) ? "not equal" : "equal") << ".\n";
        }
        std::cout << '\n';
    }


    // 1.3. Inline assembly to shift doublewords in ymm1 left by amount specified in ymm2.
    {
        std::cout << "1.3. Shift doublewords left:\n";
        alignas(sizeof(std::uint32_t) * 8)
            std::array ymm0{ 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32 };
        alignas(sizeof(std::uint32_t) * 8)
            std::array ymm1{ 2_u32, 2_u32, 2_u32, 2_u32, 2_u32, 2_u32, 2_u32, 2_u32 };
        alignas(sizeof(std::uint32_t) * 8)
            std::array ymm2{ 1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32 };
        print_array(ymm0, "ymm0   (would store result) = ");
        print_array(ymm1, "ymm1               (source) = ");
        print_array(ymm2, "ymm2           (shift mask) = ");

        asm volatile(
            "vmovaps %0, %%ymm0\n\t"  // Load data to ymm0 register.
            "vmovaps %1, %%ymm1\n\t"  // Load data to ymm1 register.
            "vmovaps %2, %%ymm2\n\t"  // Load data to ymm2 register.

            // Logical shift left doublewords in ymm1 by ymm2 and store result in ymm0.
            "vpsllvd %%ymm1, %%ymm2, %%ymm0\n\t"
            "vmovups %%ymm0, %0\n\t"  // Store data.

            : "+m"(ymm0)              // Input and output from/to ymm0 variable.
            : "m"(ymm1), "m"(ymm2));  // Input from ymm1 and ymm2 as operands.

        print_array(ymm0, "ymm0 (shifted with vpsllvd) = ");
    }


    // Task 2. You should implement 3 functions utilizing data formats __m64, __m128, __m256 from C
    // libraries (<xmmintrin.h>, <mmintrin.h>, <immintrin.h>), which perform operations on arrays.

    // 2.1. Using MMX intrinsics, perform vector addition and store results.
    {
        std::cout << "\n2.1. Add two vectors:\n";

        alignas(sizeof(std::uint16_t) * 4)
            const std::array a{ 4_u16, 3_u16, 2_u16, 1_u16 };
        alignas(sizeof(std::uint16_t) * 4)
            const std::array b{ 5_u16, 6_u16, 7_u16, 8_u16 };
        print_array(a, "a     = ");
        print_array(b, "b     = ");

        std::array<std::uint16_t, 4> result{};
        *reinterpret_cast<__m64*>(result.data()) = add_vectors(a, b);

        print_array(result, "a + b = ");
    }


    // 2.2 Using SSE intrinsics, perform square root calculation and store result.
    {
        std::cout << "\n2.2. Square root:\n";

        alignas(sizeof(double) * 2)
            const std::array a{ 16.0, 4.0 };
        print_array(a, "a       = ");

        std::array<double, 2> result{};
        *reinterpret_cast<__m128d*>(result.data()) = sqrt_intrinsic(a);

        print_array(result, "sqrt(a) = ");
    }


    // 2.3. Using AVX2 intrinsics, compare and store minimum elements
    {
        std::cout << "\n2.3. Compare and store minimum elements:\n";

        // clang-format off
        alignas(sizeof(std::uint8_t) * 32)
            const std::array a{   0_u8,    1_u8,   2_u8,     3_u8,    4_u8,    5_u8,    6_u8,    7_u8,  // min.
                                100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,
                                 16_u8,   17_u8,   18_u8,   19_u8,   20_u8,   21_u8,   22_u8,   23_u8,  // min.
                                100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8
        };
        alignas(sizeof(std::uint8_t) * 32)
            const std::array b{ 100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,
                                  8_u8,   9_u8,    10_u8,   11_u8,   12_u8,   13_u8,   14_u8,   15_u8,  // min.
                                100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,  100_u8,
                                 24_u8,  25_u8,    26_u8,   27_u8,   28_u8,   29_u8,   30_u8,   31_u8  // min.
        };
        // clang-format on

        print_array(a, "a         = ");
        print_array(b, "b         = ");

        std::array<std::uint8_t, 32> result{};
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data()), compare_vectors_min(a, b));

        print_array(result, "min(a, b) = ");
    }
}
