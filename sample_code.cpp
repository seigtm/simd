/** @file sample_code.cpp
 *  @brief ММХ-SSE-SSE2.
 *  @details This is a sample teacher's code for learning MMX-SSE-SSE2 instructions tweaked by me.
 *  @author Baranov Konstantin (seigtm) <gh@seig.ru>
 *  @version 1.0
 *  @date 2024-03-05
 */

#include <iostream>

int main() {
    // 1. Using MMX instructions to compare elements of qw1 and qw2.
    char qw1[8]{ 1, 0, 1, 1, 1, 1, 0, 1 };
    char qw2[8]{ 1, 2, 2, 1, 1, 2, 2, 1 };

    std::cout << "qw1 = ";
    for(int i{}; i < 8; ++i)
        std::cout << +qw1[i] << ' ';
    std::cout << "\nqw2 = ";
    for(int i{}; i < 8; ++i)
        std::cout << +qw2[i] << ' ';

    std::cout << "\nComparing qw1 and qw2 by using MMX instructions (0 = not equal, -1 = equal):\n";
    __asm__(
        "movq %1, %%mm0\n\t"
        "movq %2, %%mm1\n\t"
        "pcmpeqb %%mm1, %%mm0\n\t"
        "movq %%mm0, %0\n\t"
        : "=m"(qw1)
        : "m"(qw1), "m"(qw2));

    std::cout << "qw1 = ";
    for(int i{}; i < 8; ++i)
        std::cout << +qw1[i] << ' ';


    // 2. Using SSE instructions to add elements of c and d.
    float c[4]{ 1.0, 2.0, 3.0, 4.0 };
    const float d[4]{ 5.0, 6.0, 7.0, 8.0 };

    std::cout << "\n\nc = ";
    for(int i{}; i < 4; ++i)
        std::cout << c[i] << " ";
    std::cout << "\nd = ";
    for(int i{}; i < 4; ++i)
        std::cout << d[i] << " ";

    std::cout << "\nSumming elements of vectors c + d by using SSE instructions:\n";
    __asm__(
        "movups %1, %%xmm0\n\t"
        "movups %2, %%xmm1\n\t"
        "addps %%xmm1, %%xmm0\n\t"
        "movups %%xmm0, %0\n\t"
        : "=m"(c)
        : "m"(c), "m"(d));

    std::cout << "c = ";
    for(int i{}; i < 4; ++i)
        std::cout << c[i] << ' ';


    // 3. Using SSE2 instructions to compute square root of elements in f.
    double f[2]{ 16, 4 };
    std::cout << "\n\nf = " << f[0] << " " << f[1];

    std::cout << "\nComputing square root of elements in f by using SSE2 instructions:\n";
    __asm__(
        "movups %1, %%xmm1\n\t"
        "sqrtpd %%xmm1, %%xmm0\n\t"
        "movups %%xmm0, %0\n\t"
        : "=m"(f)
        : "m"(f));

    std::cout << "Square root of (f[0]) " << f[0] << " is (f[1]) " << f[1];


    // 4. Using SSE2 instructions to find the minimum of elements in a128 and b128.
    char a128[16]{ 1, 18, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31 };
    const char b128[16]{ 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31, 16 };

    std::cout << "\n\na128 = ";
    for(int i{}; i < 16; ++i)
        std::cout << +a128[i] << ' ';
    std::cout << "\nb128 = ";
    for(int i{}; i < 16; ++i)
        std::cout << +b128[i] << ' ';

    std::cout << "\nFinding minimum of elements in a128 and b128 by using SSE2 instructions:\n";
    __asm__(
        "movups %1, %%xmm0\n\t"
        "movups %2, %%xmm1\n\t"
        "pminub %%xmm1, %%xmm0\n\t"
        "movups %%xmm0, %0\n\t"
        : "=m"(a128)
        : "m"(a128), "m"(b128));

    std::cout << "Compared elements in a128 and b128: (a128[i], b128[i]) = ";
    for(int i{}; i < 16; ++i)
        std::cout << '(' << +a128[i] << ", " << +b128[i] << "); ";

    std::cout << "\nMinimum elements: a128 = ";
    for(int i{}; i < 16; ++i)
        std::cout << +a128[i] << ' ';
    std::cout << '\n';
}
