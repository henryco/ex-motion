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
        size_t pref_size;
        // ===== OCL PART =====

        // ===== KERNEL PART =====
        const int BASE_RESOLUTION = 240;
        const int color_c = 3;
        const bool linear = false;

        int n_matches = 2;
        int t_upper = 256;
        int t_lower = 2;
        int model_i = 0;
        int model_size = 50;
        int ghost_l = 2;
        int ghost_n = 300;
        int color_0 = 30;
        int lbsp_0 = 3;
        float threshold_lbsp = 0.1; // used for lbsp
        float alpha_d_min = 0.5;
        float alpha_norm = 0.5;
        float ghost_t = 0.01;
        float scale_r = 0.01;
        float t_scale_inc = 0.50;
        float t_scale_dec = 0.25;
        float v_flicker_inc = 1.0;
        float v_flicker_dec = 0.1;

        bgs::KernelType kernel_type = bgs::KERNEL_TYPE_DIAMOND_16;
        xm::ds::Color4u bgr_bg_color = xm::ds::Color4u::bgr(0, 255, 0);
        // ===== KERNEL PART =====


        // uchar:  N * w * h * [ B, G, R, LBSP_1, LBSP_2, ... ]
        ocl::Image2D bg_model;

        // float:  4 * 4: [ D_min(x), R(x), v(x), dt1-(x) ]
        ocl::Image2D utility_1;

        // short 3 * 2: [ St-1(x), T(x), Gt_acc(x) ]
        ocl::Image2D utility_2;


        bool initialized = false;
        bool ready = false;

    public:
        BgLbpSubtract();

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
    };
}

#endif //XMOTION_BG_LBP_SUBTRACT_H
