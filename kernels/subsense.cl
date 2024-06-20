
inline float xor_shift_rng(const int x, const int y, const uint seed) {
    /* Xor-shift RNGs George Marsaglia
     *  https://www.jstatsoft.org/article/download/v008i14/916
     */
    uint state = seed + x + y;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float) state / (float) UINT_MAX;
}

__kernel void kernel_subsense(
    /* Flexible Background Subtraction With Self-Balanced Local Sensitivity
     * https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf
     */
    __global const uchar *exclusion_mask,  // Single channel exclusion mask (optional, see: "exclusion" parameter)
    __global const uchar *image,           // Input image (current)  ch_n * 1
    __global const uchar *prev,            // Input image (previous) ch_n * 1
    __global uchar *bg_model,              // N * ch_n * [ B/G/R, [LBSP_1, LBSP_2] ]
    __global float *utility_1,             // 2 * 4: [ D_min(x), v(x) ]
    __global short *utility_2,             // 2 * 2: [ St-1(x), Gt(x) ]
    const ushort t_lower,                  // Lower bound for T(x) value
    const ushort t_upper,                  // Upper bound for T(x) value
    const ushort color_0,                  // Minimal color distance threshold
    const ushort lbsp_0,                   // Minimal LBSP distance threshold
    const ushort ghost_n,                  // Number of frames after foreground marked as ghost
    const ushort ghost_t,                  // Ghost threshold for local variations between It and It-1
    const ushort ghost_l,                  // Temporary low value of T(x) for pixel marked as a ghost
    const float d_min_alpha,               // Constant learning rate for D_min(x)
    const float flicker_v_inc,             // Increment v(x) value for flickering pixels
    const float flicker_v_dec,             // Decrement v(x) value for flickering pixels
    const float t_scale_inc,               // Scale for T(x) feedback increment
    const float t_scale_dec,               // Scale for T(x) feedback decrement
    const float r_scale,                   // Scale for R(x) feedback change (both directions)
    const uchar matches_req,               // Number of required matches of It(x) with B(x)
    const uchar lbsp_kernel,               // LBSP kernel type: [ 0, 1, 2, 3 ]
    const uchar model_size,                // Number of frames "N" in bg_model B(x)
    const uchar channels,                  // Number of channels in input image
    const uchar exclusion,                 // Should perform foreground exclusion using exclusion mask [0 - false]
    const uint rng_seed,                   // Seed for random number generator
    const ushort width,
    const ushort height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;


}