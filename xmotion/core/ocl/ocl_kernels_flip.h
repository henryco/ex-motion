//
// Created by henryco on 13/06/24.
//

#ifndef XMOTION_OCL_KERNELS_FLIP_H
#define XMOTION_OCL_KERNELS_FLIP_H

#include <string>
namespace xm::ocl::kernels {
    inline const std::string FLIP_ROTATE_KERNEL = R"C(
__kernel void flip_rotate(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int width,
        const int height,
        const int c_size,
        const int flip_x,
        const int flip_y,
        const int rotate
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int f_x = (flip_x > 0) ? (width  - x - 1) : x;
    const int f_y = (flip_y > 0) ? (height - y - 1) : y;
    const int idx_i = (f_y * width + f_x) * c_size;

    if (rotate > 0) {
        const int o_x = height - y - 1;
        const int idx_o = (x * height + o_x) * c_size;

        for (int i = 0; i < c_size; i++) {
            output[idx_o + i] = input[idx_i + i];
        }
        return;
    }

    const int idx_o = (y * width + x) * c_size;
    for (int i = 0; i < c_size; i++) {
        output[idx_o + i] = input[idx_i + i];
    }
}
)C";
}

#endif //XMOTION_OCL_KERNELS_FLIP_H
