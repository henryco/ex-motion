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

__kernel void kernel_bgr_to_hls(
        __global const unsigned char *input_bgr,
        __global unsigned char *out_hls,
        const unsigned int width,
        const unsigned int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char hls[3];
    const int idx = (y * width + x) * 3;
    bgr_to_hls(&input_bgr[idx], hls);

    out_hls[idx + 0] = hls[0];
    out_hls[idx + 1] = hls[1];
    out_hls[idx + 2] = hls[2];
}

__kernel void kernel_range_hls_mask(
        __global const unsigned char *input_bgr,
        __global unsigned char *output_gray,
        const unsigned int width,
        const unsigned int height,
        const unsigned char lower_h,
        const unsigned char lower_l,
        const unsigned char lower_s,
        const unsigned char upper_h,
        const unsigned char upper_l,
        const unsigned char upper_s
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const bool wrap = lower_h > upper_h;
    const int idx = y * width + x;

    unsigned char hls[3];
    bgr_to_hls(&input_bgr[idx * 3], hls);

    if (!wrap) {
        if (hls[0] >= lower_h &&
            hls[1] >= lower_l &&
            hls[2] >= lower_s &&
            hls[0] <= upper_h &&
            hls[1] <= upper_l &&
            hls[2] <= upper_s) {
            output_gray[idx] = 255;
        } else {
            output_gray[idx] = 0;
        }
    }

    else {
        if ((hls[0] >= lower_h || hls[0] <= upper_h) &&
            hls[1] >= lower_l &&
            hls[2] >= lower_s &&
            hls[1] <= upper_l &&
            hls[2] <= upper_s) {
            output_gray[idx] = 255;
        } else {
            output_gray[idx] = 0;
        }
    }
}