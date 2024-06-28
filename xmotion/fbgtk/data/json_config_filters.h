//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_FILTERS_H
#define XMOTION_JSON_CONFIG_FILTERS_H

#include "json_config_common.h"

namespace xm::data {

    typedef std::string FilterType;
#define XM_FILTER_TYPE_BLUR "blur"
#define XM_FILTER_TYPE_DIFF "difference"
#define XM_FILTER_TYPE_CHROMA "chromakey"

    namespace fbg {
        enum BgKernelType {
            CROSS_4 = 1,
            SQUARE_8 = 2,
            RUBY_12 = 3,
            DIAMOND_16 = 4
        };
    }

    typedef struct {
        std::string key; // hex color
        std::string replace; // hex color
        HSL range;
        int blur;
        int power;
        int fine;
        int refine;
        bool linear;
        bool _present;
    } Chroma;

    typedef struct Difference {
        int BASE_RESOLUTION = 240;       // Segmentation mask base resolution (px)
        std::string color = "#ffffff";   // New background color

        bool debug_on = false;           // Should enable debug functions
        bool adapt_on = true;            // Should enable updates of background model B(x)
        bool ghost_on = true;            // Should enable "ghost" detection
        bool lbsp_on = true;             // Should use Local Binary Similarity Patterns for spatial comparison
        bool norm_l2 = true;             // Should use L2 distance (and norm) for color comparison
        bool linear = false;             // Should use linear interpolation for image downscaling

        float color_0 = 0.032;           // threshold used in color comparison above which pixel is classified as different
        float lbsp_0 = 0.06;             // threshold used in lbsp  comparison above which pixel is classified as different
        float lbsp_d = 0.025;            // threshold used in lbsp calculation

        int n_matches = 2;               // number of intersections of I(x) with B(x) to detect background
        int t_upper = 256;               // Maximal value of T(x), higher T(x) -> lower p
        int t_lower = 2;                 // Minimal value of T(x), lower T(x) -> higher p
        int model_size = 50;             // Number of frames in B(x), frame consist of N color pixels (BGR) with LBSP string for each of them
        int ghost_l = 2;                 // Temporary new T(x) value for pixel classified as a "ghost"
        int ghost_n = 300;               // Number of frames for which pixel is unchanged to be classified as a ghost
        int ghost_n_inc = 1;             // Increment value for ghost_n accumulator (see "ghost_n")
        int ghost_n_dec = 15;            // Decrement value for ghost_n accumulator (see "ghost_n")
        float alpha_d_min = 0.75;        // Constant learning rate for D_min(x): [ D_min(x) =   dt(x) * a + D_min(x) * (1-a) ]
        float alpha_norm = 0.75;         // Mixing alpha for dt(x) calculation:  [ dt(x)    = d_color * a + d_lbsp   * (1-a) ]
        float ghost_t = 0.25;            // Ghost threshold for local variations (dt(x)) between It and It-1
        float r_scale = 0.1;             // Scale for R(x) feedback change (both directions)
        float r_cap = 255;               // Max value for R(x)
        float t_scale_inc = 0.50;        // Scale for T(x) feedback increment
        float t_scale_dec = 0.25;        // Scale for T(x) feedback decrement
        float v_flicker_inc = 1.0;       // Increment v(x) value for flickering pixels
        float v_flicker_dec = 0.1;       // Decrement v(x) value for flickering pixels
        float v_flicker_cap = 255;       // Maximum   v(x) value for flickering pixels

        int refine_gate = 0;             // Number of Gate     operation to be applied
        int refine_erode = 0;            // Number of Erosion  operation to be applied
        int refine_dilate = 0;           // Number of Dilation operation to be applied

        float gate_threshold = 0.85;     // Gate operation threshold

        fbg::BgKernelType kernel = fbg::DIAMOND_16;
        fbg::BgKernelType gate_kernel = fbg::DIAMOND_16;
        fbg::BgKernelType erode_kernel = fbg::RUBY_12;
        fbg::BgKernelType dilate_kernel = fbg::RUBY_12;

        bool _present = false;
    } Difference;

    typedef struct {
        int power;
        bool _present;
    } Blur;

    // Yes, I'm not going to practice polymorphic type gymnastics here
    typedef struct {
        xm::data::FilterType type;
        Blur blur;
        Chroma chroma;
        Difference difference;
    } Filter;
}

#endif //XMOTION_JSON_CONFIG_FILTERS_H
