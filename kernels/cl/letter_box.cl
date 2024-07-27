inline void downscale(
    __global const uchar *in_img,          // from
    uchar *out_pix,                        // to
    const ushort img_w,                    // from
    const ushort img_h,                    // from
    const float scale_w,                   // [from : to], ie: [2 : 1]
    const float scale_h,                   // [from : to], ie: [2 : 1]
    const int pix_x,                       // to
    const int pix_y,                       // to
    const uchar channels_n,                // number of color channels, ie: 1/2/3/4
    const bool linear                      // linear or nearest interpolation
) {
    if (!linear) {
        const int img_x = clamp((int) (pix_x * scale_w), (int) 0, (int) (img_w - 1));
        const int img_y = clamp((int) (pix_y * scale_h), (int) 0, (int) (img_h - 1));
        const int pos = ((img_y * img_w) + img_x) * channels_n;
        for (int i = 0; i < channels_n; i++)
            out_pix[i] = in_img[pos + i];
        return;
    }

    const float img_x = pix_x * scale_w;
    const float img_y = pix_y * scale_h;

    const int x0 = (int) img_x;
    const int y0 = (int) img_y;
    const float dx = img_x - x0;
    const float dy = img_y - y0;

    const int x1 = min((int) (x0 + 1), (int) (img_w - 1));
    const int y1 = min((int) (y0 + 1), (int) (img_h - 1));

    for (int i = 0; i < channels_n; i++) {
        float p00 = in_img[(y0 * img_w + x0) * 3 + i];
        float p10 = in_img[(y0 * img_w + x1) * 3 + i];
        float p01 = in_img[(y1 * img_w + x0) * 3 + i];
        float p11 = in_img[(y1 * img_w + x1) * 3 + i];

        float p0 = p00 + (p10 - p00) * dx;
        float p1 = p01 + (p11 - p01) * dx;
        float p = p0 + (p1 - p0) * dy;
        out_pix[i] = (unsigned char) p;
    }
}

__kernel void kernel_downscale_letterbox(
    __global const uchar *image_bgr,       // Input  image [From] (larger)
    __global       float *output_rgb,      // Output image [To]   (smaller)
             const ushort img_w,           // From width
             const ushort img_h,           // From height
             const ushort out_w,           // Output width
             const ushort out_h,           // Output height
             const ushort padding_w,       // Horizontal padding
             const ushort padding_h,       // Vertical padding
             const float scale_w,          // [From : To], ie: [2 : 1]
             const float scale_h,          // [From : To], ie: [2 : 1]
             const uchar channels_n,       // Number of color channels, ie: 1/2/3
             const uchar linear            // is linear interpolation used, [0 - false]
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= out_w || y >= out_h)
        return;

    const int idx = (y * out_w + x) * channels_n;

    if (x < padding_w || x >= out_w - padding_w
    ||  y < padding_h || y >= out_h - padding_h) {
        for (int i = 0; i < channels_n; i++)
            output_rgb[idx + i] = 0.f;
        return;
    }

    uchar color_pixel[3]; // bgr
    downscale(
        image_bgr,
        color_pixel,
        img_w,
        img_h,
        scale_w,
        scale_h,
        x - padding_w,
        y - padding_h,
        channels_n,
        linear > 0);

    for (int i = 0; i < channels_n; i++) // rgb
        output_rgb[idx + i] = (float) color_pixel[channels_n - i - 1] / 255.f;
}