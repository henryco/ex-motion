
typedef uchar LbspKernelType;
#define LBSP_KERNEL_NONE       0
#define LBSP_KERNEL_CROSS_4    1
#define LBSP_KERNEL_SQUARE_8   2
#define LBSP_KERNEL_DIAMOND_16 3

#define L2_C3_NORM_DIV 441.6729559.f
#define L2_C2_NORM_DIV 360.6244584.f
#define L2_C1_NORM_DIV 255.f

inline float xor_shift_rng(int x, int y, uint seed) {
    /* Xor-shift RNGs George Marsaglia
     *  https://www.jstatsoft.org/article/download/v008i14/916
     */
    uint state = seed + x + y;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float) state / (float) UINT_MAX;
}

inline int lbsp_k_size_bytes(LbspKernelType t) {
    if (t == LBSP_KERNEL_NONE)
        return 0;
    if (t == LBSP_KERNEL_DIAMOND_16)
        return 2;
    return 1;
}

inline int pos_3(int x, int y, int z, int w, int h, int c_sz) {
    return ((z * h + y) * w + x) * c_sz;
}

inline int hamming_distance(const uchar *one, const uchar *two, int n) {
    int d = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char c = one[i] ^ two[i];
        for (int j = 0; j < 8; j++)
            d += ((c >> j) & 1);
    }
    return d;
}

inline int l1_distance(const uchar *one, const uchar *two, int n) {
    int d = 0;
    for (int i = 0; i < n; i++)
        d += abs((int) one[i] - (int) two[i]);
    return d;
}

inline float l2_distance(const uchar *one, const uchar *two, int n) {
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

inline float normalize_hd(int value, LbspKernelType t) {
    return t == LBSP_KERNEL_DIAMOND_16
        ? (float) value / 16.f
        : t == LBSP_KERNEL_SQUARE_8
            ? (float) value / 8.f
            : (float) value / 4.f;
}

void compute_lbsp(
        const uchar *input,
        uchar *out,
        const LbspKernelType kernel_type,
        const uchar threshold,
        const int channels_n,
        const int width,
        const int height,
        const int x,
        const int y
) {
    const int offset = kernel_type == LBSP_KERNEL_CROSS_4
        ? 24
        : kernel_type == LBSP_KERNEL_SQUARE_8
            ? 16
            : 0;

    const char KER_ARR[32] = {
        -2, -2,
        -2,  2,
         2, -2,
         2,  2,
        -2,  0
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

      --- == --- offset  0

      O   O   O
        . . .
      O . X . O
        . . .
      O   O   O

    ----- + ----- offset 16

        O . O
        . X .
        O . O

    ----- + ----- offset 24

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
            const int i_x = x + KER_ARR[i    ];
            const int i_y = y + KER_ARR[i + 1];

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

__kernel void kernel_subsense(
    /* Flexible Background Subtraction With Self-Balanced Local Sensitivity
     * https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf
     */

#ifndef DISABLED_EXCLUSION_MASK
    __global const uchar *exclusion_mask,  // Single channel exclusion mask (optional, see: "exclusion" parameter)
    const uchar exclusion,                 // Should perform foreground exclusion using exclusion mask [0 - false]
#endif

    __global const uchar *image,           // Input image (current)  ch_n * 1
    __global const uchar *prev,            // Input image (previous) ch_n * 1
    __global uchar *bg_model,              // N * ch_n * [ B, G, R, LBSP_1, LBSP_2, ... ]:
    __global float *utility_1,             // 3 * 4: [ D_min(x), R(x), v(x) ]
    __global short *utility_2,             // 2 * 2: [ St-1(x), T(x), Gt_acc(x) ]
    __global uchar *seg_mask,              // Output segmentation mask St(x)

#ifndef DISABLED_LBSP
    const uchar lbsp_kernel,               // LBSP kernel type: [ 0, 1, 2, 3 ]
    const uchar lbsp_threshold,            // Threshold value used for initial LBSP calculation
    const float n_norm_alpha,              // Normalization weight factor between color and lbsp distance [0...1]
    const ushort lbsp_0,                   // Minimal LBSP distance threshold for pixels to be marked as different
#endif

    const ushort color_0,                  // Minimal color distance threshold for pixels to be marked as different
    const ushort t_lower,                  // Lower bound for T(x) value
    const ushort t_upper,                  // Upper bound for T(x) value
    const ushort ghost_n,                  // Number of frames after foreground marked as ghost ( see also Gt_acc(x) )
    const ushort ghost_t,                  // Ghost threshold for local variations between It and It-1
    const ushort ghost_l,                  // Temporary low value of T(x) for pixel marked as a ghost
    const float d_min_alpha,               // Constant learning rate for D_min(x) [0...1]
    const float flicker_v_inc,             // Increment v(x) value for flickering pixels
    const float flicker_v_dec,             // Decrement v(x) value for flickering pixels
    const float t_scale_inc,               // Scale for T(x) feedback increment
    const float t_scale_dec,               // Scale for T(x) feedback decrement
    const float r_scale,                   // Scale for R(x) feedback change (both directions)
    const uchar matches_req,               // Number of required matches of It(x) with B(x)
    const uchar model_size,                // Number of frames "N" in bg_model B(x)
    const uchar channels_n,                // Number of color channels in input image [1, 2, 3]
    const uint rng_seed_1,                  // Seed for random number generator
    const uint rng_seed_2,                  // Seed for random number generator
    const ushort width,
    const ushort height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

#ifndef DISABLED_EXCLUSION_MASK
    // 0. Test for exclusion mask
    if (exclusion > 0 && exclusion_mask[idx] > 0) {
        seg_mask[idx] = 255;
        return; // this is foreground from exclusion mask
    }
#endif

    const float random_value = xor_shift_rng(x, y, rng_seed_1);

    const int img_idx = idx * channels_n;
    const int ut1_idx = idx * 3;
    const int ut2_idx = idx * 2;

    const float D_m = utility_1[ut1_idx    ];
    const float R_x = utility_1[ut1_idx + 1];
    const float V_x = utility_1[ut1_idx + 2];

    const bool St_1 = utility_2[ut2_idx] > 0;
    const short T_x = utility_2[ut2_idx + 1];
    const short G_c = utility_2[ut2_idx + 2];

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
        const int bgm_idx = pos_3(x, y, i, width, height, bgm_ch_size);

#ifndef DISABLED_LBSP
        const int d_lbsp  = hamming_distance(lbsp_img, &bg_model[bgm_idx + channels_n], kernel_size);
        const float d_l_n = normalize_hd(d_lbsp, lbsp_kernel);
#endif

        const int d_color = l1_distance(&image[img_idx], &bg_model[bgm_idx], channels_n);
        const float d_c_n = normalize_l1(d_color, channels_n);

#ifndef DISABLED_LBSP
        const float dtx = n_norm_alpha * d_c_n + n_norm_alpha_inv * d_l_n;
#else
        const float dtx = d_c_n;
#endif

        D_MIN_X = min(D_MIN_X, dtx);

        if (d_color > color_0
#ifndef DISABLED_LBSP
            && d_lbsp > lbsp_0
#endif
        ) {
            if (++matches >= matches_req) {
                // Foreground detected
                is_foreground = true;
                seg_mask[idx] = 255;
                break;
            }
        }

        if (++i >= model_size)
            i = 0;
    }

    // update moving average D_min(x)
    const float new_D_m = D_m * (1.f - d_min_alpha) + dtx * d_min_alpha;
    utility_1[ut1_idx]  = new_D_m;

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
    const short new_T_x = clamp(
        (is_foreground
            ? T_x + t_scale_inc * (1.f / (new_V_x * new_D_m))
            : T_x - t_scale_dec * (new_V_x / new_D_m)
        ),
        t_lower,
        t_upper);
    utility_2[ut2_idx + 1] = new_T_x;
    // TODO update GHOSTS

    // update St-1(x)
    utility_2[ut2_idx    ] = is_foreground ? 255 : 0;

    // update B(x)
    if (random_value <= 1 / new_T_x) {
        const int random_frame_n   = (float) xor_shift_rng(x, y, rng_seed_1) * (model_size - 1);
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