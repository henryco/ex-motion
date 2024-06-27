//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_BG_LBP_SUBTRACT_H
#define XMOTION_BG_LBP_SUBTRACT_H

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
            KERNEL_TYPE_DIAMOND_16 = 3,
        };

        int lbsp_k_size_bytes(KernelType t);

        using Conf = struct Conf {
            /**
             * New background color (BGR)
             */
            xm::ds::Color4u color;

            /**
             * Local Binary Patterns kernel type
             */
            KernelType kernel_type = KERNEL_TYPE_DIAMOND_16;

            /**
             * Mask refinement iterations
             */
            int refine = 0;
        };
    }

    class BgLbpSubtract : public xm::Filter {
        static inline const auto log =
                spdlog::stdout_color_mt("filter_bg_lbp_subtract");

    private:
        // ===== OCL PART =====
        std::map<int, cl_command_queue> ocl_queue_map;
        cl_command_queue ocl_command_queue;
        cl_device_id device_id;
        cl_context ocl_context;
        cl_program program_subsense;
        cl_kernel kernel_apply;
        cl_kernel kernel_prepare;
        cl_kernel kernel_subsense;
        cl_kernel kernel_downscale;
        cl_kernel kernel_upscale;
        cl_kernel kernel_dilate;
        cl_kernel kernel_erode;
        cl_kernel kernel_debug;
        size_t pref_size;
        // ===== OCL PART =====


        // ===== KERNEL PART =====
        const int BASE_RESOLUTION = 240;
        const int color_c = 3;
        const bool debug_on = true;
        const bool ghost_on = true;
        const bool lbsp_on = true;
        const bool norm_l2 = true;
        const bool mask_xc = false;
        const bool linear = false;

        float color_0 = 0.069;
        float lbsp_0 = 0.31;

        int n_matches = 2;
        int t_upper = 256;
        int t_lower = 2;
        int model_i = 0;
        int model_size = 50;
        int ghost_l = 2;
        int ghost_n = 300;
        int ghost_n_inc = 1;
        int ghost_n_dec = 15;
        float threshold_lbsp = 0.0025; // used for lbsp
        float alpha_d_min = 0.75;
        float alpha_norm = 0.75;
        float ghost_t = 0.25;
        float r_scale = 0.01;
        float r_cap = 255;
        float t_scale_inc = 0.50;
        float t_scale_dec = 0.25;
        float v_flicker_inc = 1.0;
        float v_flicker_dec = 0.1;
        float v_flicker_cap = 100;

        bgs::KernelType kernel_type = bgs::KERNEL_TYPE_DIAMOND_16;
        xm::ds::Color4u bgr_bg_color = xm::ds::Color4u::bgr(255, 255, 255);
        // ===== KERNEL PART =====


        // ===== REFINE PART =====
        int refine_erode = 0;
        int refine_dilate = 0;
        bgs::KernelType erode_type = bgs::KERNEL_TYPE_SQUARE_8;
        bgs::KernelType dilate_type = bgs::KERNEL_TYPE_SQUARE_8;
        // ===== REFINE PART =====


        // uchar:  N * w * h * [ B, G, R, LBSP_1, LBSP_2, ... ]
        ocl::Image2D bg_model;

        // float:  4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
        ocl::Image2D utility_1;

        // short 3 * 2: [ St-1(x), T(x), Gt_acc(x) ]
        ocl::Image2D utility_2;

        // float: w * h
        ocl::Image2D noise_map;

        bool initialized = false;
        bool ready = false;

    public:
        BgLbpSubtract() = default;

        ~BgLbpSubtract() override;

        void init(const bgs::Conf &conf);

        xm::ocl::iop::ClImagePromise filter(const ocl::iop::ClImagePromise &in, int q_idx) override;

        void reset() override;

        void start() override;

        void stop() override;

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
    };
}

#endif //XMOTION_BG_LBP_SUBTRACT_H
