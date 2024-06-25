
typedef uchar MorphType;
#define MORPH_TYPE_ERODE  0
#define MORPH_TYPE_DILATE 1

typedef uchar KernelType;
#define KERNEL_TYPE_NONE       0
#define KERNEL_TYPE_CROSS_4    1
#define KERNEL_TYPE_SQUARE_8   2
#define KERNEL_TYPE_DIAMOND_16 3

#define L2_C3_NORM_DIV 441.6729559f
#define L2_C2_NORM_DIV 360.6244584f
#define L2_C1_NORM_DIV 255.f

inline float xor_shift_rng(const uint seed) {
    /* Xor-shift RNGs George Marsaglia
     *  https://www.jstatsoft.org/article/download/v008i14/916
     */
    uint state = seed;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float) state / (float) UINT_MAX;
}

inline int lbsp_k_size_bytes(KernelType t) {
#ifndef DISABLED_LBSP
    if (t == KERNEL_TYPE_NONE)
        return 0;
    if (t == KERNEL_TYPE_DIAMOND_16)
        return 2;
    return 1;
#else
    return 0;
#endif
}

inline int pos_3(int x, int y, int z, int w, int h, int c_sz) {
    return ((z * h + y) * w + x) * c_sz;
}

inline int hamming_distance(
    __global const uchar *one,
             const uchar *two,
                   int n
) {
    int d = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char c = one[i] ^ two[i];
        for (int j = 0; j < 8; j++)
            d += ((c >> j) & 1);
    }
    return d;
}

inline int l1_distance(
    __global const uchar *one,
    __global const uchar *two,
                   int n
) {
    int d = 0;
    for (int i = 0; i < n; i++)
        d += abs((int) one[i] - (int) two[i]);
    return d;
}

inline float l2_distance(
    __global const uchar *one,
    __global const uchar *two,
                   int n
) {
    float d = 0;
    for (int i = 0; i < n; i++)
        d += pow((float) one[i] - (float) two[i], 2.f);
    return sqrt(d);
}

inline float normalize_l1(int value, int channels_n) {
    return (float) value / (255.f * (float) channels_n);
}

inline float normalize_l2_3(float value) {
    return value / L2_C3_NORM_DIV; // sqrt(3 * 255^2)
}

inline float normalize_l2_2(float value) {
    return value / L2_C2_NORM_DIV; // sqrt(2 * 255^2)
}

inline float normalize_l2_1(float value) {
    return value /L2_C1_NORM_DIV;
}

inline float normalize_l2(float value, int channels_n) {
    return channels_n == 3
        ? normalize_l2_3(value)
        : channels_n == 2
            ? normalize_l2_2(value)
            : normalize_l2_1(value);
}

inline float normalize_hd(int value, int channels_n, KernelType t) {
    return t == KERNEL_TYPE_DIAMOND_16
        ? (float) value / (channels_n * 16.f)
        : t == KERNEL_TYPE_SQUARE_8
            ? (float) value / (channels_n * 8.f)
            : (float) value / (channels_n * 4.f);
}

void compute_lbsp(
    __global const uchar *input,
    uchar *out,
    const KernelType kernel_type,
    const uchar threshold,
    const int channels_n,
    const int width,
    const int height,
    const int x,
    const int y
) {
    const int offset = kernel_type == KERNEL_TYPE_CROSS_4
        ? 24
        : kernel_type == KERNEL_TYPE_SQUARE_8
            ? 16
            : 0;

    const char KER_ARR[32] = {
        -2, -2,
        -2,  2,
         2, -2,
         2,  2,
        -2,  0,
         0, -2,
         2,  0,
         0,  2,

        -1, -1,
        -1,  1,
         1, -1,
         1,  1,

        -1,  0,
         0, -1,
         1,  0,
         0,  1
    };

    /*

        O   O   O
          O O O
        O O X O O
          O O O
        O   O   O

    ------- = ------- offset  0

        O   O   O
          . . .
        O . X . O
          . . .
        O   O   O

    ------- + ------- offset 16

          O . O
          . X .
          O . O

    ------- + ------- offset 24

            O
          O X O
            O

    */

    out[0] = 0;
    int c = 0, b = 0;

    const int idx_mid = (y * width + x) * channels_n;

    for (int i = 0; i < channels_n; i++) {
        const int mid = input[idx_mid + i] + threshold; // B, G or R with min threshold

        for (int h = offset; h < 31; h += 2) {
            const int i_x = x + KER_ARR[h    ];
            const int i_y = y + KER_ARR[h + 1];

            if (i_x >= 0 && i_y >= 0 && i_x < width && i_y < height) {
                if (input[((i_y * width) + i_x) * channels_n + i] > mid)
                    out[c] |= ((unsigned char) 1) << b;
            }
            if (++b >= 8) {
                out[++c] = 0;
                b = 0;
            }
        }
    }
}

void downscale(
    __global const uchar *in_img,    // from
    uchar *out_pix,                  // to
    const ushort img_w,              // from
    const ushort img_h,              // from
    const float scale_w,             // [from : to], ie: [2 : 1]
    const float scale_h,             // [from : to], ie: [2 : 1]
    const int pix_x,                 // to
    const int pix_y,                 // to
    const uchar channels_n,          // number of color channels, ie: 1/2/3/4
    const bool linear                // linear or nearest interpolation
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

void upscale(
    __global const uchar *input_pix,          // Input color pixel [From]
    __global uchar *output,             // Output image [To]
    const ushort output_w,           // To width
    const ushort output_h,           // To height
    const float scale_w,             // [to : from], ie: [2 : 1]
    const float scale_h,             // [to : from], ie: [2 : 1]
    const uchar ceil_scale_w,        // ceil(scale_w)
    const uchar ceil_scale_h,        // ceil(scale_h)
    const uchar channels_n,          // number of color channels, ie: 1/2/3/4
    const int pix_x,                 // From x
    const int pix_y                  // From y
) {
    const float img_x = pix_x * scale_w;
    const float img_y = pix_y * scale_h;

    for (int ky = 0; ky < ceil_scale_h; ky++) {
        for (int kx = 0; kx < ceil_scale_w; kx++) {
            const int ix = clamp((int) (img_x + kx), (int) 0, (int) (output_w - 1));
            const int iy = clamp((int) (img_y + ky), (int) 0, (int) (output_h - 1));
            const int o_idx = (iy * output_w + ix) * channels_n;

            for (int i = 0; i < channels_n; i++)
                output[o_idx + i] = input_pix[i];
        }
    }
}

void upscale_apply(
             const uchar *color_pix,          // Color value for background area
    __global const uchar *input,              // Input image (larger)
    __global       uchar *output,             // Output image [To] (larger)
             const ushort output_w,           // To width
             const ushort output_h,           // To height
             const float scale_w,             // [to : from], ie: [2 : 1]
             const float scale_h,             // [to : from], ie: [2 : 1]
             const uchar ceil_scale_w,        // ceil(scale_w)
             const uchar ceil_scale_h,        // ceil(scale_h)
             const uchar channels_n,          // number of color channels, ie: 1/2/3/4
             const uchar mask,                // Background segmentation mask
             const int pix_x,                 // From x
             const int pix_y                  // From y
) {
    const float img_x = pix_x * scale_w;
    const float img_y = pix_y * scale_h;

    for (int ky = 0; ky < ceil_scale_h; ky++) {
        for (int kx = 0; kx < ceil_scale_w; kx++) {
            const int ix = clamp((int) (img_x + kx), (int) 0, (int) (output_w - 1));
            const int iy = clamp((int) (img_y + ky), (int) 0, (int) (output_h - 1));
            const int o_idx = (iy * output_w + ix) * channels_n;

            if (mask > 0) {
                // foreground
                for (int i = 0; i < channels_n; i++)
                    output[o_idx + i] = input[o_idx + i];
            } else {
                // background
                for (int i = 0; i < channels_n; i++)
                    output[o_idx + i] = color_pix[i];
            }
        }
    }
}

void morph_operation(
    __global const uchar *input,  // Input image
    __global       uchar *output, // Output image
    const MorphType morph_type,   // Erode - 0, Dilate - 1, see (MorphType)
    const KernelType kernel_type, // Kernel type: [ 0, 1, 2, 3 ], see (KernelType)
    const uchar channels_n,       // Number of color channels, ie: 1/2/3/4
    const ushort width,           // Image width
    const ushort height,          // Image height
    const int x,
    const int y
) {
    const int img_idx = (y * width + x) * channels_n;
    const char KER_ARR[32] = {
        -2, -2,
        -2,  2,
        2, -2,
        2,  2,
        -2,  0,
        0, -2,
        2,  0,
        0,  2,

        -1, -1,
        -1,  1,
        1, -1,
        1,  1,

        -1,  0,
        0, -1,
        1,  0,
        0,  1
    };
    const int offset = kernel_type == KERNEL_TYPE_CROSS_4
        ? 24
        : kernel_type == KERNEL_TYPE_SQUARE_8
            ? 16
            : 0;

    for (int i = 0; i < channels_n; i++) {
        uchar value = input[img_idx + i];

        for (int h = offset; h < 31; h += 2) {
            const int i_x = x + KER_ARR[h    ];
            const int i_y = y + KER_ARR[h + 1];

            if (i_x < 0 || i_x >= width || i_y < 0 || i_y >= height)
                continue;

            if (morph_type == MORPH_TYPE_DILATE) {
                value = max(value, input[(i_y * width + i_x) * channels_n + i]);
                continue;
            }

            // MORPH_TYPE_ERODE default
            value = min(value, input[(i_y * width + i_x) * channels_n + i]);
        }
        output[img_idx + i] = value;
    }
}

__kernel void kernel_subsense(
    /* Flexible Background Subtraction With Self-Balanced Local Sensitivity
     * https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf
     */

#ifndef DISABLED_EXCLUSION_MASK
    __global const uchar *exclusion_mask,  // Single channel exclusion mask (optional, see: "exclusion" parameter)
             const uchar exclusion,        // Should perform foreground exclusion using exclusion mask [0 - false]
#endif

    __global const uchar *image,           // Input image (current)  ch_n * 1
    __global       uchar *bg_model,        // N * ch_n * [ B, G, R, LBSP_1, LBSP_2, ... ]
    __global       float *utility_1,       // 4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
    __global       short *utility_2,       // 3 * 2: [ St-1(x), T(x), Gt_acc(x) ]
    __global       uchar *seg_mask,        // Output segmentation mask St(x)

#ifndef DISABLED_LBSP
             const uchar lbsp_kernel,      // LBSP kernel type: [ 0, 1, 2, 3 ], see (KernelType)
             const uchar lbsp_threshold,   // Threshold value used for initial LBSP calculation
             const float n_norm_alpha,     // Normalization weight factor between color and lbsp distance [0...1]
             const ushort lbsp_0,          // Minimal LBSP distance threshold for pixels to be marked as different
#endif

             const ushort color_0,         // Minimal color distance threshold for pixels to be marked as different
             const ushort t_lower,         // Lower bound for T(x) value
             const ushort t_upper,         // Upper bound for T(x) value
             const ushort ghost_n,         // Number of frames after foreground marked as ghost ( see also Gt_acc(x) )
             const ushort ghost_l,         // Temporary low value of T(x) for pixel marked as a ghost
             const float ghost_t,          // Ghost threshold for local variations between It and It-1
             const float d_min_alpha,      // Constant learning rate for D_min(x) [0...1]
             const float flicker_v_inc,    // Increment v(x) value for flickering pixels
             const float flicker_v_dec,    // Decrement v(x) value for flickering pixels
             const float t_scale_inc,      // Scale for T(x) feedback increment
             const float t_scale_dec,      // Scale for T(x) feedback decrement
             const float r_scale,          // Scale for R(x) feedback change (both directions)
             const uchar matches_req,      // Number of required matches of It(x) with B(x)
             const uchar model_size,       // Number of frames "N" in bg_model B(x)
             const uchar channels_n,       // Number of color channels in input image [1, 2, 3]
             const uint rng_seed,          // Seed for random number generator
             const ushort width,
             const ushort height

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

#ifndef DISABLED_EXCLUSION_MASK
    // Test for exclusion mask
    if (exclusion > 0 && exclusion_mask[idx] > 0) {
        seg_mask[idx] = 255;
        return; // this is foreground from exclusion mask
    }
#endif

    const float random_value = xor_shift_rng(idx + rng_seed);

    const int img_idx = idx * channels_n;
    const int ut1_idx = idx * 4;
    const int ut2_idx = idx * 3;

    const float D_m = utility_1[ut1_idx    ];
    const float R_x = utility_1[ut1_idx + 1];
    const float V_x = utility_1[ut1_idx + 2];

    const bool St_1 = utility_2[ut2_idx] > 0;
    const short T_x = utility_2[ut2_idx + 1];

#ifndef DISABLED_LBSP
    const float n_norm_alpha_inv = 1.f - n_norm_alpha;
    const int kernel_size = lbsp_k_size_bytes(lbsp_kernel);
    const int bgm_ch_size = channels_n * (1 + kernel_size);
    const int r_lbsp  = (int) (pow(2.f, R_x) + lbsp_0);
    uchar lbsp_img[6]; // (2 bytes = 16 bits) x (B+G+R = 3)
    compute_lbsp(image, lbsp_img, lbsp_kernel, lbsp_threshold, channels_n, width, height, x, y);
#else
    const int bgm_ch_size = channels_n;
#endif

    const int bg_model_start = (int) (random_value * (model_size - 1));
    const int r_color = (int) (R_x * color_0);

    bool is_foreground = false;
    float D_MIN_X = 1.f;
    int matches = 0;

    for (int i = bg_model_start, k = model_size; k >= 0; k--) {
        const int bgm_idx   = pos_3(x, y, i, width, height, bgm_ch_size);

#ifndef DISABLED_LBSP
        const int d_lbsp    = hamming_distance(&bg_model[bgm_idx + channels_n], lbsp_img, kernel_size);
        const float d_l_n   = normalize_hd(d_lbsp, channels_n, lbsp_kernel);
#endif

#ifndef COLOR_NORM_l2
        const int d_color   = l1_distance(&image[img_idx], &bg_model[bgm_idx], channels_n);
        const float d_c_n   = normalize_l1(d_color, channels_n);
#else
        const float d_color = l2_distance(&image[img_idx], &bg_model[bgm_idx], channels_n);
        const float d_c_n   = normalize_l2(d_color, channels_n);
#endif

#ifndef DISABLED_LBSP
        const float dtx = n_norm_alpha * d_c_n + n_norm_alpha_inv * d_l_n;
#else
        const float dtx = d_c_n;
#endif

        D_MIN_X = min(D_MIN_X, dtx);

        if (d_color > r_color // maybe replace with floating normalized threshold?
#ifndef DISABLED_LBSP
          && d_lbsp > r_lbsp  // maybe replace with floating normalized threshold? // TODO FIXME
#endif
        ) {
            if (++matches >= matches_req) {
                // Foreground detected
                is_foreground = true;
                break;
            }
        }

        if (++i >= model_size)
            i = 0;
    }

    // update out segmentation mask St(x)
    seg_mask[idx] = is_foreground ? 255 : 0;

    // update G_c(x)
    const float new_G_c = is_foreground && (fabs(utility_1[ut1_idx + 3] - D_MIN_X) < ghost_t)
        ? utility_2[ut2_idx + 2] + 1
        : 0;
    utility_2[ut2_idx + 2] = new_G_c;

    // update moving average D_min(x) and dt-1(x)
    const float new_D_m = D_m * (1.f - d_min_alpha) + D_MIN_X * d_min_alpha;
    utility_1[ut1_idx    ] = new_D_m;
    utility_1[ut1_idx + 3] = D_MIN_X;

    // update v(x)
    const float new_V_x = is_foreground != St_1
        ? (V_x + flicker_v_inc)
        : max(flicker_v_dec, V_x - flicker_v_dec);
    utility_1[ut1_idx + 2] = new_V_x;

    // update R(x)
    utility_1[ut1_idx + 1] = R_x < pow(1.f + new_D_m * 2.f, 2.f)
        ? R_x + r_scale * (new_V_x - flicker_v_dec) // that " -flicker_v_dec " is heuristic, whatever
        : max(1.f, R_x - (r_scale / new_V_x));

    // update T(x)
    short new_T_x = clamp(
        (ushort) (is_foreground
            ? T_x + t_scale_inc * (1.f / (new_V_x * new_D_m))
            : T_x - t_scale_dec * (new_V_x / new_D_m)
        ),
        t_lower,
        t_upper);
    utility_2[ut2_idx + 1] = new_T_x;

    // Ghost detected
    if (new_G_c > ghost_n) {
        new_T_x = ghost_l;
    }

    // update St-1(x)
    utility_2[ut2_idx] = is_foreground ? 255 : 0;

    // update B(x)
    if (!is_foreground && random_value <= 1 / new_T_x) {
        // Don't ask why: xor_shift_rng(T_x + img_idx + rng_seed), just a simple heuristic
        const int random_frame_n   = (float) xor_shift_rng(T_x + img_idx + rng_seed) * (model_size - 1);
        const int random_frame_idx = pos_3(x, y, random_frame_n, width, height, bgm_ch_size);
        for (int i = 0; i < channels_n; i++){
            bg_model[random_frame_idx + i] = image[img_idx + i]; // B, G, R
        }
#ifndef DISABLED_LBSP
        const int random_frame_idx_offset = random_frame_idx + channels_n;
        for (int i = 0; i < kernel_size; i++) {
            bg_model[random_frame_idx_offset + i] = lbsp_img[i]; // LBSP bit strings
        }
#endif
    }

}

__kernel void kernel_prepare_model(

    __global const uchar *image,           // Input image (current)  ch_n * 1
    __global       uchar *bg_model,        // N:     [ B, G, R, LBSP_1, LBSP_2, ... ]
    __global       float *utility_1,       // 4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
    __global       short *utility_2,       // 3 * 2: [ St-1(x), T(x), Gt_acc(x) ]

#ifndef DISABLED_LBSP
             const uchar lbsp_kernel,      // LBSP kernel type: [ 0, 1, 2, 3 ], see (KernelType)
             const uchar lbsp_threshold,   // Threshold value used for initial LBSP calculation
#endif
             const uchar model_i,          // Current model index (z)
             const uchar model_size,       // Number of frames "N" in bg_model B(x)
             const uchar channels_n,       // Number of color channels in input image [1, 2, 3]
             const ushort t_lower,         // Lower bound for T(x) value
             const ushort width,
             const ushort height

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

#ifndef DISABLED_LBSP
    const int kernel_size = lbsp_k_size_bytes(lbsp_kernel);
    const int bgm_ch_size = channels_n * (1 + kernel_size);
    uchar lbsp_img[6]; // (2 bytes = 16 bits) x (B+G+R = 3)
    compute_lbsp(image, lbsp_img, lbsp_kernel, lbsp_threshold, channels_n, width, height, x, y);
#else
    const int bgm_ch_size = channels_n;
#endif

    const int idx     = y * width + x;
    const int ut1_idx = idx * 4;
    const int ut2_idx = idx * 3;
    const int img_idx = idx * channels_n;
    const int bgm_idx = pos_3(x, y, clamp((int) model_i, 0, model_size - 1), width, height, bgm_ch_size);

    utility_1[ut1_idx    ] = .5f;       // D_min(x)
    utility_1[ut1_idx + 1] = 1;         // R(x)
    utility_1[ut1_idx + 2] = 1;         // v(x)
    utility_1[ut1_idx + 3] = .5f;       // dt-1(x)

    utility_2[ut2_idx    ] = 0;         // St-1(x)
    utility_2[ut2_idx + 1] = t_lower;   // T(x)
    utility_2[ut2_idx + 2] = 0;         // Gt_acc(x)

    for (int i = 0; i < channels_n; i++) {
        bg_model[bgm_idx + i] = image[img_idx + i]; // B, G, R
    }

#ifndef DISABLED_LBSP
    const int bgm_idx_offset = bgm_idx + channels_n;
    for (int i = 0; i < kernel_size; i++) {
        bg_model[bgm_idx_offset + i] = lbsp_img[i]; // LBSP bit strings
    }
#endif
}

__kernel void kernel_downscale(

    __global const uchar *image,           // Input  image [From] (larger)
    __global       uchar *output,          // Output image [To]   (smaller)
             const ushort img_w,           // From width
             const ushort img_h,           // From height
             const ushort out_w,           // To   width
             const ushort out_h,           // To   height
             const float scale_w,          // [From : To], ie: [2 : 1]
             const float scale_h,          // [From : To], ie: [2 : 1]
             const uchar channels_n,       // Number of color channels, ie: 1/2/3/4
             const uchar linear            // is linear interpolation used, [0 - false]

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= out_w || y >= out_h)
        return;

    uchar color_pixel[4];
    downscale(image, color_pixel, img_w, img_h, scale_w, scale_h, x, y, channels_n, linear > 0);

    const int idx = (y * out_w + x) * channels_n;
    for (int i = 0; i < channels_n; i++)
        output[idx + i] = color_pixel[i];
}

__kernel void kernel_upscale(

    __global const uchar *image,           // Input  image [From] (smaller)
    __global       uchar *output,          // Output image [To]   (larger)
             const ushort img_w,           // From width
             const ushort img_h,           // From height
             const ushort out_w,           // To   width
             const ushort out_h,           // To   height
             const float scale_w,          // [To : From], ie: [2 : 1]
             const float scale_h,          // [To : From], ie: [2 : 1]
             const uchar d_x,              // ceil(scale_w)
             const uchar d_y,              // ceil(scale_h)
             const uchar channels_n        // Number of color channels, ie: 1/2/3/4

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= img_w || y >= img_h)
        return;

    upscale(
        &image[(y * img_w + x) * channels_n], output,
        out_w, out_h,
        scale_w, scale_h,
        d_x, d_y,
        channels_n,
        x, y);
}

__kernel void kernel_upscale_apply(

    __global const uchar *mask,            // Segmentation mask [From] (smaller) 1 channel
    __global const uchar *image,           // Source image      [To]   (larger)  N channels
    __global       uchar *output,          // Output image      [To]   (larger)  N channels
             const ushort mask_w,          // From width
             const ushort mask_h,          // From height
             const ushort out_w,           // To   width
             const ushort out_h,           // To   height
             const float scale_w,          // [To : From], ie: [2 : 1]
             const float scale_h,          // [To : From], ie: [2 : 1]
             const uchar d_x,              // ceil(scale_w)
             const uchar d_y,              // ceil(scale_h)
             const uchar color_b,          // New background color B
             const uchar color_g,          // New background color G
             const uchar color_r,          // New background color R
             const uchar channels_n        // Number of color channels, ie: 1/2/3/4

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= mask_w || y >= mask_h)
        return;

    const uchar colors[3] = {
        color_b,
        color_g,
        color_r
    };

    upscale_apply(
        colors, image, output,
        out_w, out_h,
        scale_w, scale_h,
        d_x, d_y,
        channels_n,
        mask[y * mask_w + x],
        x, y);
}

__kernel void kernel_erode(

    __global const uchar *input,           // Input image
    __global       uchar *output,          // Output image
             const uchar kernel_type,      // Kernel type: [ 0, 1, 2, 3 ], see (KernelType)
             const uchar channels_n,       // Number of color channels, ie: 1/2/3/4
             const ushort width,           // Image width
             const ushort height           // Image height

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    morph_operation(input, output, MORPH_TYPE_ERODE, kernel_type, channels_n, width, height, x, y);
}

__kernel void kernel_dilate(

    __global const uchar *input,           // Input image
    __global       uchar *output,          // Output image
             const uchar kernel_type,      // Kernel type: [ 0, 1, 2, 3 ], see (KernelType)
             const uchar channels_n,       // Number of color channels, ie: 1/2/3/4
             const ushort width,           // Image width
             const ushort height           // Image height

) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    morph_operation(input, output, MORPH_TYPE_DILATE, kernel_type, channels_n, width, height, x, y);
}

__kernel void kernel_debug(
    __global const uchar *bg_model,        // N:     [ B, G, R, LBSP_1, LBSP_2, ... ]
    __global const float *utility_1,       // 4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
    __global const short *utility_2,       // 3 * 2: [ St-1(x), T(x), Gt_acc(x) ]
    __global       uchar *output,          // 3 * 1: [ B, G, R ]
             const uchar lbsp_kernel,      // LBSP kernel type: [ 0, 1, 2, 3 ], see (KernelType)
             const uchar model_size,       // Number of frames "N" in bg_model B(x)
             const uchar select_n,         // Debug type
             const uint rng_seed,          // Seed for random number generator
             const ushort width,
             const ushort height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx     = y * width + x;
    const int ut1_idx = idx * 4;
    const int ut2_idx = idx * 3;
    const int img_idx = idx * 3;

    const float random_value = xor_shift_rng(idx + rng_seed);
    const int k_size_b = lbsp_k_size_bytes(lbsp_kernel);
    const int c_size_b = 3 + (3 * k_size_b);

    // Model last color
    if (select_n == 0) {
        const int z = model_size - 1;
        const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = bg_model[bgm_idx + i];
        return;
    }

    // Model random frame color
    if (select_n == 1) {
        const int z = clamp((int) (random_value * (model_size - 1)), 0, model_size - 1);
        const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = bg_model[bgm_idx + i];
        return;
    }

    // Model average colors
    if (select_n == 2) {
        int color_avg[3] = {0, 0, 0};
        for (int z = 0; z < model_size - 1; z++) {
            const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
            for (int i = 0; i < 3; i++)
                color_avg[i] += bg_model[bgm_idx + i];
        }
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = (int) ((float) color_avg[i] / (float) model_size);
        return;
    }

#ifndef DISABLED_LBSP
    // Model last LBSP
    if (select_n == 3) {
        const int z = model_size - 1;
        const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
        const int offset_idx = bgm_idx + 3;
        const uchar ref[6] = {0, 0};
        const int d = hamming_distance(&bg_model[offset_idx], ref, k_size_b);
        const float nd = normalize_hd(d, 3, lbsp_kernel);
        const int nc = (int) (nd * 255.f);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = nc;
        return;
    }

    // Model random LBSP
    if (select_n == 4) {
        const int z = clamp((int) (random_value * (model_size - 1)), 0, model_size - 1);
        const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
        const int offset_idx = bgm_idx + 3;
        const uchar ref[6] = {0, 0};
        const int d = hamming_distance(&bg_model[offset_idx], ref, k_size_b);
        const float nd = normalize_hd(d, 3, lbsp_kernel);
        const int nc = (int) (nd * 255.f);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = nc;
        return;
    }

    // Model average LBSP
    if (select_n == 5) {
        int avg = 0;

        for (int z = 0; z < model_size - 1; z++) {
            const int bgm_idx = pos_3(x, y, z, width, height, c_size_b);
            const int offset_idx = bgm_idx + 3;
            const uchar ref[6] = {0, 0};
            const int d = hamming_distance(&bg_model[offset_idx], ref, k_size_b);
            const float nd = normalize_hd(d, 3, lbsp_kernel);
            avg += (int) (nd * 255.f);
        }

        avg = (int) ((float) avg / (float) model_size);

        for (int i = 0; i < 3; i++)
            output[img_idx + i] = avg;
        return;
    }
#endif

    // D_min(x)
    if (select_n == 6) {
        const float d_x = utility_1[ut1_idx];
        int v = (int) (d_x * 255.f);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = v;
        return;
    }

    // dt-1(x)
    if (select_n == 7) {
        const float d_x = utility_1[ut1_idx + 3];
        int v = (int) (d_x * 255.f);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = v;
        return;
    }

    // R(x)
    if (select_n == 8) {
        const float v = utility_1[ut1_idx + 1];
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = v * 10.f;
        return;
    }

    // v(x)
    if (select_n == 9) {
        const int v = clamp((int) utility_1[ut1_idx + 2], 0, 255);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = v;
        return;
    }

    // St-1(x)
    if (select_n == 10) {
        const short v = utility_2[ut2_idx];
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = (uchar) v;
        return;
    }

    // T(x)
    if (select_n == 11) {
        const int v = clamp((int) utility_2[ut2_idx + 1], 0, 255);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = (uchar) v;
        return;
    }

    // Gt_acc(x)
    if (select_n == 12) {
        const int v = clamp((int) utility_2[ut2_idx + 2], 0, 255);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = (uchar) v;
        return;
    }

    // random value
    if (select_n == 13) {
        const int v = (int) (random_value * 255.f);
        for (int i = 0; i < 3; i++)
            output[img_idx + i] = v;
    }
}
