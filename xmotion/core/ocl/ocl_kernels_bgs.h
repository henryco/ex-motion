//
// Created by henryco on 14/06/24.
//

#ifndef XMOTION_OCL_KERNELS_BGS_H
#define XMOTION_OCL_KERNELS_BGS_H

#include <string>

namespace xm::ocl::kernels {

    inline const std::string COMPUTE_LBP = R"ocl(
inline int hamming_distance(
        const unsigned char *one,
        const unsigned char *two,
        const int n
) {
    int d = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char c = one[i] ^ two[i];
        for (int j = 0; j < 8; j++)
            d += ((c >> j) & 1);
    }
    return d;
}

inline void compute_lbp(
        const unsigned char *input,
        unsigned char* out,
        const int c_input_size,  // numbers of color channels in image
        const int c_code_size,   // ceil(pow(kernel_size, 2) / 8.f)
        const int kernel_size,   // actually should be <= 15
        const int width,
        const int height,
        const int x,
        const int y
) {
    out[0] = 0;

    unsigned char mid = 0;
    const int idx_mid = (y * width + x) * c_input_size;
    for (int i = 0; i < c_input_size; i++)
        mid += input[idx_mid + i];
    mid /= c_input_size;

    int c = 0, b = 0;
    for (int ky = -kernel_size; ky <= kernel_size; ky++) {
        for (int kx = -kernel_size; kx <= kernel_size; kx++) {
            if (ky == 0 || kx == 0)
                continue;

            const int i_x = x + kx;
            const int i_y = y + ky;

            unsigned char p = 0;
            const int idx_p = (i_y * width + i_x) * c_input_size;
            for (int i = 0; i < c_input_size; i++)
                p += input[idx_p + i];
            p /= c_input_size;

            const bool value = i_x >= 0
                    && i_y >= 0
                    && i_x < width
                    && i_y < height
                    && p >= mid;

            out[c] |= value << b;

            b++;
            if (b >= 8) {
                b = 0;
                c++;
                out[c] = 0;
            }
        }
    }
}

inline unsigned char mask_lbp(
        __global const unsigned char *input_bgr,
        __global const unsigned char *static_lbp,
        const int c_input_size,  // numbers of color channels in image
        const int c_code_size,   // ceil(pow(kernel_size, 2) / 8.f)
        const int kernel_size,   // actually should be <= 15
        const int total_bits,    // pow(kernel_size, 2) - 1
        const float threshold,   // [0 ... 1]
        const int width,
        const int height,
        const int x,
        const int y
) {
    unsigned char lbp[32]; // up to 256 bits
    compute_lbp(input_bgr, lbp, c_input_size, c_code_size, kernel_size, width, height, x, y);

    const int idx_s = (y * width + x) * c_code_size;
    const int d = hamming_distance(lbp, &(static_lbp[idx_s]), c_code_size);

    return ((float) d / (float) total_bits) >= threshold ? 255 : 0;
}

__kernel void kernel_lbp(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int c_input_size,  // numbers of color channels in image
        const int c_code_size,   // ceil(pow(kernel_size, 2) / 8.f)
        const int kernel_size,   // actually should be <= 15
        const int width,
        const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char lbp[32]; // up to 256 bits
    compute_lbp(input, lbp, c_input_size, c_code_size, kernel_size, width, height, x, y);

    const int idx = (y * width + x) * c_code_size;
    for (int i = 0; i < c_code_size; i++)
        output[idx + i] = lbp[i];
}

)ocl";

}

#endif //XMOTION_OCL_KERNELS_BGS_H
