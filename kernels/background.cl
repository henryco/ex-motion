
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
        const unsigned char *frame_image,
        const unsigned char *frame_lbp,
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
    compute_lbp(frame_image, lbp, c_input_size, kernel_size, width, height, x, y);

    const int idx_s = (y * width + x) * c_code_size;
    const int d = hamming_distance(lbp, &(frame_lbp[idx_s]), c_code_size);

    return ((float) d / (float) total_bits) >= threshold ? 255 : 0;
}

__kernel void kernel_lbp(
        __global const unsigned char *frame_image,
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
    compute_lbp(frame_image, lbp, c_input_size, kernel_size, width, height, x, y);

    const int idx = (y * width + x) * c_code_size;
    for (int i = 0; i < c_code_size; i++)
        output[idx + i] = lbp[i];
}


__kernel void kernel_mask_only(
        __global const unsigned char *frame_image,
        __global const unsigned char *frame_lbp,
        __global unsigned char *output_mask_bin,
        const int c_input_size,  // numbers of color channels in image
        const int c_code_size,   // ceil(pow(kernel_size, 2) / 8.f)
        const int kernel_size,   // actually should be <= 15
        const int total_bits,    // pow(kernel_size, 2) - 1
        const float threshold,   // [0 ... 1]
        const int width,
        const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    output_mask_bin[y * width + x] = mask_lbp(
            frame_image,
            frame_lbp,
            c_input_size,
            c_code_size,
            kernel_size,
            total_bits,
            threshold,
            width,
            height,
            x, y);
}

__kernel void kernel_mask_apply(
        __global const unsigned char *frame_image,
        __global const unsigned char *mask_bw,
        __global unsigned char* output,
        const int c_input_size,  // numbers of color channels in image
        const int width,
        const int height,
        const unsigned char color_b,
        const unsigned char color_g,
        const unsigned char color_r
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx_raw = y * width + x;
    const int idx_pix = idx_raw * c_input_size;
    const unsigned char mask = mask_bw[idx_raw];

    if (mask > 0) {
        for (int i = 0; i < c_input_size; i++)
            output[idx_pix + i] = frame_image[idx_pix + i];
        return;
    }

    const unsigned char colors[3] = { color_b, color_g, color_r };
    for (int i = 0; i < c_input_size; i++)
        output[idx_pix + i] = colors[i];
}

__kernel void kernel_lbp_mask_apply(
        __global const unsigned char *frame_image,
        __global const unsigned char *frame_lbp,
        __global unsigned char* output,
        const int c_input_size,  // numbers of color channels in image
        const int c_code_size,   // ceil(pow(kernel_size, 2) / 8.f)
        const int kernel_size,   // actually should be <= 15
        const int total_bits,    // pow(kernel_size, 2) - 1
        const float threshold,   // [0 ... 1]
        const int width,
        const int height,
        const unsigned char color_b,
        const unsigned char color_g,
        const unsigned char color_r
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const unsigned char mask = mask_lbp(
        frame_image,
        frame_lbp,
        c_input_size,
        c_code_size,
        kernel_size,
        total_bits,
        threshold,
        width,
        height,
        x, y);

    const int idx = (y * width + x) * c_input_size;
    if (mask > 0) {
        for (int i = 0; i < c_input_size; i++)
            output[idx + i] = frame_image[idx + i];
        return;
    }

    const unsigned char colors[3] = { color_b, color_g, color_r };
    for (int i = 0; i < c_input_size; i++)
        output[idx + i] = colors[i];
}