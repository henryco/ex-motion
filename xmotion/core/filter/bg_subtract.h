//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_BG_SUBTRACT_H
#define XMOTION_BG_SUBTRACT_H

#include "../utils/xm_data.h"
#include "i_filter.h"

#include <map>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::filters {

    void new_size(int w, int h, int base, int &new_w, int &new_h, float &scale);

    uint32_t time_seed();

    namespace bgs {
        enum KernelType {
            KERNEL_TYPE_NONE = 0,
            KERNEL_TYPE_CROSS_4 = 1,
            KERNEL_TYPE_SQUARE_8 = 2,
            KERNEL_TYPE_RUBY_12 = 3,
            KERNEL_TYPE_DIAMOND_16 = 4,
        };

        int lbsp_k_size_bytes(KernelType t);

        using Conf = struct Conf {
            int BASE_RESOLUTION = 240;   // Segmentation mask base resolution (px)
            int color_channels = 3;      // Number of color channels in image

            bool adapt_on = true;        // Should enable updates of background model B(x)
            bool debug_on = true;        // Should enable debug functions
            bool morph_on = true;        // Should enable morphological operations (erode/dilate/etc.)
            bool ghost_on = true;        // Should enable "ghost" detection
            bool lbsp_on = true;         // Should use Local Binary Similarity Patterns for spatial comparison
            bool norm_l2 = true;         // Should use L2 distance (and norm) for color comparison
            bool mask_xc = false;        // Should use early exclusion mask
            bool linear = false;         // Should use linear interpolation for image downscaling

            float color_0 = 0.032;       // threshold used in color comparison above which pixel is classified as different
            float lbsp_0 = 0.06;         // threshold used in lbsp  comparison above which pixel is classified as different
            float lbsp_d = 0.025;        // threshold used in lbsp calculation

            int n_matches = 2;           // number of intersections of I(x) with B(x) to detect background
            int t_upper = 256;           // Maximal value of T(x), higher T(x) -> lower p
            int t_lower = 2;             // Minimal value of T(x), lower T(x) -> higher p
            int model_size = 50;         // Number of frames in B(x), frame consist of N color pixels (BGR) with LBSP string for each of them
            int ghost_l = 2;             // Temporary new T(x) value for pixel classified as a "ghost"
            int ghost_n = 300;           // Number of frames for which pixel is unchanged to be classified as a ghost
            int ghost_n_inc = 1;         // Increment value for ghost_n accumulator (see "ghost_n")
            int ghost_n_dec = 15;        // Decrement value for ghost_n accumulator (see "ghost_n")
            float alpha_d_min = 0.75;    // Constant learning rate for D_min(x): [ D_min(x) =   dt(x) * a + D_min(x) * (1-a) ]
            float alpha_norm = 0.75;     // Mixing alpha for dt(x) calculation:  [ dt(x)    = d_color * a + d_lbsp   * (1-a) ]
            float ghost_t = 0.25;        // Ghost threshold for local variations (dt(x)) between It and It-1
            float r_scale = 0.1;         // Scale for R(x) feedback change (both directions)
            float r_cap = 255;           // Max value for R(x)
            float t_scale_inc = 0.50;    // Scale for T(x) feedback increment
            float t_scale_dec = 0.25;    // Scale for T(x) feedback decrement
            float v_flicker_inc = 1.0;   // Increment v(x) value for flickering pixels
            float v_flicker_dec = 0.1;   // Decrement v(x) value for flickering pixels
            float v_flicker_cap = 255;   // Maximum   v(x) value for flickering pixels

            bgs::KernelType kernel = bgs::KERNEL_TYPE_DIAMOND_16;
            xm::ds::Color4u color = xm::ds::Color4u::bgr(255, 255, 255);

            // ===== REFINE PART =====
            int refine_gate = 0;
            int refine_erode = 0;
            int refine_dilate = 0;

            float refine_gate_threshold = 0.85;

            bgs::KernelType gate_kernel = bgs::KERNEL_TYPE_DIAMOND_16;
            bgs::KernelType erode_kernel = bgs::KERNEL_TYPE_RUBY_12;
            bgs::KernelType dilate_kernel = bgs::KERNEL_TYPE_RUBY_12;
            // ===== REFINE PART =====
        };
    }

    class BgSubtract : public xm::Filter {
        static inline const auto log =
                spdlog::stdout_color_mt("filter_bg_lbp_subtract");

    private:
        // ===== OCL PART =====
        std::map<int, cl_command_queue> ocl_queue_map;
        cl_command_queue ocl_command_queue = nullptr;
        cl_device_id device_id = nullptr;
        cl_context ocl_context = nullptr;
        cl_program program_subsense = nullptr;
        cl_kernel kernel_apply = nullptr;
        cl_kernel kernel_prepare = nullptr;
        cl_kernel kernel_subsense = nullptr;
        cl_kernel kernel_downscale = nullptr;
        cl_kernel kernel_upscale = nullptr;
        cl_kernel kernel_dilate = nullptr;
        cl_kernel kernel_erode = nullptr;
        cl_kernel kernel_gate = nullptr;
        cl_kernel kernel_debug = nullptr;
        size_t pref_size = 0;
        // ===== OCL PART =====

        // uchar:  N * w * h * [ B, G, R, LBSP_1, LBSP_2, ... ]
        ocl::Image2D bg_model;

        // float:  4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
        ocl::Image2D utility_1;

        // short 2 * 2: [ T(x), Gt_acc(x) ]
        ocl::Image2D utility_2;

        // float: w * h
        ocl::Image2D noise_map;

        // uchar: w * h
        ocl::Image2D seg_mask;

        // uchar: w * h
        ocl::Image2D tmp_mask;

        bgs::Conf config;

        bool initialized = false;
        bool ready = false;

        int debug_mode = -1;
        int model_i = 0;

    public:
        BgSubtract() = default;

        ~BgSubtract() override;

        void init(const bgs::Conf &conf);

        xm::ocl::iop::ClImagePromise filter(const ocl::iop::ClImagePromise &in, int q_idx) override;

        xm::ocl::iop::ClImagePromise filter(const ocl::iop::ClImagePromise &in, const ocl::iop::ClImagePromise &ex_mask, int q_idx);

        void reset() override;

        void start() override;

        void stop() override;

        void set_debug_mode(int mode);

    protected:
        cl_command_queue retrieve_queue(int index);

        void prepare_update_model(const ocl::iop::ClImagePromise &frame_in, int q_idx);

        xm::ocl::iop::ClImagePromise downscale(const ocl::iop::ClImagePromise &in, int base, int q_idx);

        xm::ocl::iop::ClImagePromise subsense(const ocl::iop::ClImagePromise &downscaled,
                                              const ocl::iop::ClImagePromise &original,
                                              const ocl::iop::ClImagePromise &exclusion, // optional
                                              int q_idx);

        void release();

        xm::ocl::iop::ClImagePromise debug(int n, const xm::ocl::iop::ClImagePromise &ref);

        int denorm_color_threshold(float v) const;

        int denorm_lbsp_threshold(float v) const;

        void erode(cl_command_queue queue, size_t *l_size, size_t *g_size);

        void dilate(cl_command_queue queue, size_t *l_size, size_t *g_size);

        void gate(cl_command_queue queue, size_t *l_size, size_t *g_size);
    };
}

#endif //XMOTION_BG_SUBTRACT_H
