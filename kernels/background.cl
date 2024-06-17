
inline void bgr_to_hls(
        const unsigned char *in_bgr,
        unsigned char *out_hls
) {
    // https://docs.opencv.org/4.9.0/de/d25/imgproc_color_conversions.html
    const float b = ((float) in_bgr[0]) / 255.f;
    const float g = ((float) in_bgr[1]) / 255.f;
    const float r = ((float) in_bgr[2]) / 255.f;

    const float c_max = fmax(b, fmax(g, r));
    const float c_min = fmin(b, fmin(g, r));
    const float c_dif = c_max - c_min;
    const float c_sum = c_max + c_min;
    const float c_fac = 60.f / c_dif;
    const float L = (c_sum / 2.f);

    float H = 0.f;
    if (c_max == r)
        H = c_fac * (g - b) ;
    else if (c_max == g)
        H = 120.f + (c_fac * (b - r));
    else if (c_max == b)
        H = 240.f + (c_fac * (r - g));
    if (H < 0)
        H += 360.f;

    out_hls[0] = (unsigned char) (H * 0.708333f); // 255 / 360
    out_hls[1] = (unsigned char) (L * 255.f);
    out_hls[2] = (unsigned char) ((L < 0.5f ? (c_dif / c_sum) : (c_dif / (2.f - c_sum))) * 255.f);
}

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
        unsigned char *out,
        const int c_input_size,  // numbers of color channels in image
        const int kernel_size,   // actually should be <= 15
        const int width,
        const int height,
        const int x,
        const int y
) {
    out[0] = 0;

    int mid = 0;
    const int idx_mid = (y * width + x) * c_input_size;
    for (int i = 0; i < c_input_size; i++)
        mid += (int) input[idx_mid + i];
    mid /= c_input_size;

    int c = 0, b = 0;
    for (int ky = -kernel_size; ky <= kernel_size; ky++) {
        for (int kx = -kernel_size; kx <= kernel_size; kx++) {
            if (ky == 0 || kx == 0)
                continue;

            const int i_x = x + kx;
            const int i_y = y + ky;

            if (i_x >= 0 && i_y >= 0 && i_x < width && i_y < height) {
                int p = 0;
                const int idx_p = (i_y * width + i_x) * c_input_size;
                for (int i = 0; i < c_input_size; i++)
                    p += (int) input[idx_p + i];
                p /= c_input_size;

                if (p >= mid) {
                    out[c] |= ((unsigned char) 1) << b;
                }
            }

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
        const int c_code_size,   // ceil((pow(kernel_size, 2) - 1) / 8.f)
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
        __global unsigned char *output,
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

__kernel void kernel_color_diff(
        __global const unsigned char *frame_img,
        __global const unsigned char *frame_ref,
        __global unsigned char *output,
        const int c_input_size,  // numbers of color channels in image
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

    const int idx = (y * width + x) * c_input_size;

    unsigned char hls_img[3];
    unsigned char hls_ref[3];

    bgr_to_hls(&frame_img[idx], hls_img);
    bgr_to_hls(&frame_ref[idx], hls_ref);

    const int d_abs = abs(hls_img[0] - hls_ref[0]);
    const int d_hue = min(d_abs, (int)255 - d_abs);

    const int d_sat = abs(hls_img[2] - hls_ref[2]);

    const int diff = max(d_hue, d_sat);

    for (int i = 0; i < c_input_size; i++) {
        output[idx + i] = diff;
    }
}