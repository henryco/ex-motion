//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/filter/bg_subtract.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/ocl/cl_kernel.h"
#include "../../kernels/subsense.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
#pragma ide diagnostic ignored "ConstantConditionsOC"
namespace xm::filters {

    BgSubtract::~BgSubtract() {
        release();
    }

    void BgSubtract::release() {
        for (auto &item: ocl_queue_map) {
            if (item.second == nullptr)
                continue;
            clReleaseCommandQueue(item.second);
        }

        if (kernel_apply != nullptr) clReleaseKernel(kernel_apply);
        if (kernel_prepare != nullptr) clReleaseKernel(kernel_prepare);
        if (kernel_subsense != nullptr) clReleaseKernel(kernel_subsense);
        if (kernel_downscale != nullptr) clReleaseKernel(kernel_downscale);
        if (kernel_upscale != nullptr) clReleaseKernel(kernel_upscale);
        if (kernel_dilate != nullptr) clReleaseKernel(kernel_dilate);
        if (kernel_erode != nullptr) clReleaseKernel(kernel_erode);
        if (kernel_gate != nullptr) clReleaseKernel(kernel_gate);
        if (kernel_debug != nullptr) clReleaseKernel(kernel_debug);
        if (program_subsense != nullptr) clReleaseProgram(program_subsense);
        if (ocl_command_queue != nullptr) clReleaseCommandQueue(ocl_command_queue);
        if (ocl_context != nullptr) clReleaseContext(ocl_context);
        if (device_id != nullptr) clReleaseDevice(device_id);
    }

    cl_command_queue BgSubtract::retrieve_queue(int index) {
        if (index <= 0)
            return ocl_command_queue;

        if (ocl_queue_map.contains(index))
            return ocl_queue_map.at(config.debug_on);

        ocl_queue_map.emplace(index, xm::ocl::create_queue_device(
                ocl_context,
                device_id,
                true,
                false));
        return ocl_queue_map[index];
    }

    void BgSubtract::init(const bgs::Conf &conf) {
        config = conf;

        reset();
        release();

        device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        ocl_command_queue = xm::ocl::create_queue_device(
            ocl_context,
            device_id,
            true,
            false);

        std::string options = std::string("")
            + (config.norm_l2 ? " -DCOLOR_NORM_l2 " : "")
            + (config.mask_xc ? "" : " -DDISABLED_EXCLUSION_MASK ")
            + (config.lbsp_on ? "" : " -DDISABLED_LBSP ")
            + (config.debug_on ? "" : " -DDISABLED_DEBUG ")
            + (config.ghost_on ? "" : " -DDISABLED_GHOST ")
            + (config.adapt_on ? "" : " -DDISABLED_ADAPT ")
            + (config.morph_on ? "" : " -DDiSABLED_MORPH ")
            ;

        program_subsense = xm::ocl::build_program(
            ocl_context, device_id,
            ocl_kernel_subsense_data,
            ocl_kernel_subsense_data_size,
            "subsense.cl",
            options
        );

        kernel_apply = xm::ocl::build_kernel(program_subsense, "kernel_upscale_apply");
        kernel_prepare = xm::ocl::build_kernel(program_subsense, "kernel_prepare_model");
        kernel_subsense = xm::ocl::build_kernel(program_subsense, "kernel_subsense");
        kernel_downscale = xm::ocl::build_kernel(program_subsense, "kernel_downscale");
        kernel_upscale = xm::ocl::build_kernel(program_subsense, "kernel_upscale");

        if (config.morph_on) {
            kernel_dilate = xm::ocl::build_kernel(program_subsense, "kernel_dilate");
            kernel_erode = xm::ocl::build_kernel(program_subsense, "kernel_erode");
            kernel_gate = xm::ocl::build_kernel(program_subsense, "kernel_gate_mask");
        }

        if (config.debug_on)
            kernel_debug = xm::ocl::build_kernel(program_subsense, "kernel_debug");

        pref_size = xm::ocl::optimal_local_size(device_id, kernel_subsense);

        initialized = true;
    }

    xm::ocl::iop::ClImagePromise BgSubtract::filter(const ocl::iop::ClImagePromise &frame_in, int q_idx) {
        return filter(frame_in, {}, q_idx);
    }

    xm::ocl::iop::ClImagePromise BgSubtract::filter(const ocl::iop::ClImagePromise &frame_in,
                                                    const ocl::iop::ClImagePromise &ex_mask,
                                                    int q_idx) {
        if (!initialized)
            throw std::logic_error("Filter is not initialized");

        if (!ready)
            return frame_in;

        auto downscaled = downscale(frame_in, config.BASE_RESOLUTION, q_idx);

        if (model_i < config.model_size) {
            prepare_update_model(downscaled, q_idx);
            model_i += 1;
            return frame_in;
        }

        auto result = subsense(downscaled, frame_in, ex_mask, q_idx);
        return config.debug_on && debug_mode >= 0 ? debug(debug_mode, result) : result;
    }

    void BgSubtract::prepare_update_model(const ocl::iop::ClImagePromise &in_p, int q_idx) {
        cl_command_queue queue = q_idx < 0 && in_p.queue() != nullptr ? in_p.queue() : retrieve_queue(q_idx);
        const auto &in = in_p.getImage2D();

        const int n_w = (int) in.cols;
        const int n_h = (int) in.rows;

        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) n_w, pref_size),
                            xm::ocl::optimal_global_size((int) n_h, pref_size)};

        const int lbsp_c_size = config.lbsp_on ? bgs::lbsp_k_size_bytes(config.kernel) : 0;
        if (bg_model.empty()) {
            bg_model = xm::ocl::Image2D::allocate(
                n_w, n_h, (size_t) config.model_size * (config.color_channels + (config.color_channels * lbsp_c_size)), 1,
                ocl_context, device_id);
        }

        if (utility_1.empty()) {
            utility_1 = xm::ocl::Image2D::allocate(
                n_w, n_h, config.debug_on ? 5 : 4, sizeof(float),
                ocl_context, device_id);
        }

        if (utility_2.empty()) {
            utility_2 = xm::ocl::Image2D::allocate(
                n_w, n_h, 2, sizeof(short),
                ocl_context, device_id);
        }

        if (noise_map.empty()) {
            noise_map = xm::ocl::Image2D::allocate(
                    n_w, n_h, 1, sizeof(float),
                    ocl_context, device_id);
        }

        if (seg_mask.empty()) {
            seg_mask = xm::ocl::Image2D::allocate(
                    n_w, n_h, 1, 1,
                    ocl_context, device_id);
        }

        if (tmp_mask.empty()) {
            tmp_mask = xm::ocl::Image2D::allocate(
                    n_w, n_h, 1, 1,
                    ocl_context, device_id);
        }

        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.get_handle(ocl::ACCESS::RO);
        cl_mem buffer_noise = (cl_mem) noise_map.handle;
        cl_mem buffer_seg_mask = (cl_mem) seg_mask.handle;
        cl_mem buffer_bg_model = (cl_mem) bg_model.handle;
        cl_mem buffer_utility1 = (cl_mem) utility_1.handle;
        cl_mem buffer_utility2 = (cl_mem) utility_2.handle;

        auto lbsp_kernel = (uchar) config.kernel;
        auto lbsp_threshold = (uchar) std::min(255.f, config.lbsp_d * 255.f);
        auto _model_i = (uchar) model_i;
        auto _model_size = (uchar) config.model_size;
        auto _channels_n = (uchar) config.color_channels;
        auto _t_higher = (ushort) config.t_upper;
        auto _flicker_v_dec = (float) config.v_flicker_dec;
        auto _width = (ushort) n_w;
        auto _height = (ushort) n_h;

        cl_uint idx_1 = 0;
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_in);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_noise);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_seg_mask);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_bg_model);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_utility1);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_utility2);

        if (config.lbsp_on) {
            idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &lbsp_kernel);
            idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &lbsp_threshold);
        }

        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_model_i);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_model_size);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_channels_n);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(ushort), &_t_higher);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(float), &_flicker_v_dec);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(ushort), &_width);
        xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(ushort), &_height);

        xm::ocl::enqueue_kernel_fast(
            queue,
            kernel_prepare,
            2,
            g_size,
            l_size,
            false);
    }

    xm::ocl::iop::ClImagePromise BgSubtract::downscale(const ocl::iop::ClImagePromise &in_p, int base, int q_idx) {
        cl_command_queue queue = q_idx < 0 && in_p.queue() != nullptr ? in_p.queue() : retrieve_queue(q_idx);
        const auto &in = in_p.getImage2D();

        float scale;
        int n_w, n_h;

        new_size((int) in.cols, (int) in.rows, base, n_w, n_h, scale);
        const int inter_size = n_w * n_h * config.color_channels * (int) sizeof(char);
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) n_w, pref_size),
                            xm::ocl::optimal_global_size((int) n_h, pref_size)};

        cl_int err;

        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.get_handle(ocl::ACCESS::RO);
        cl_mem buffer_io_1 = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, inter_size, NULL, &err);

        auto img_w = (ushort) in.cols;
        auto img_h = (ushort) in.rows;
        auto out_w = (ushort) n_w;
        auto out_h = (ushort) n_h;
        auto scale_w = (float) scale;
        auto scale_h = (float) scale;
        auto channels_n = (uchar) config.color_channels;
        auto is_linear = (uchar) config.linear ? 255 : 0;

        cl_uint idx_0 = 0;
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(cl_mem), &buffer_in);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(cl_mem), &buffer_io_1);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(ushort), &img_w);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(ushort), &img_h);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(ushort), &out_w);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(ushort), &out_h);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(float), &scale_w);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(float), &scale_h);
        idx_0 = xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(uchar), &channels_n);
        xm::ocl::set_kernel_arg(kernel_downscale, idx_0, sizeof(uchar), &is_linear);

        xm::ocl::enqueue_kernel_fast(
            queue,
            kernel_downscale,
            2,
            g_size,
            l_size,
            false);

        return xm::ocl::iop::ClImagePromise(
            xm::ocl::Image2D(
                n_w, n_h, config.color_channels, sizeof(uchar),
                buffer_io_1, ocl_context, device_id),
                queue)
                .withCleanup(in_p);
    }

    xm::ocl::iop::ClImagePromise BgSubtract::subsense(const ocl::iop::ClImagePromise &downscaled_p,
                                                      const ocl::iop::ClImagePromise &original_p,
                                                      const ocl::iop::ClImagePromise &exclusion_p,
                                                      int q_idx) {
        cl_command_queue queue = q_idx < 0 && downscaled_p.queue() != nullptr
            ? downscaled_p.queue()
            : retrieve_queue(q_idx);

        const auto image = downscaled_p.getImage2D();
        const auto original = original_p.getImage2D();

        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) image.cols, pref_size),
                            xm::ocl::optimal_global_size((int) image.rows, pref_size)};

        cl_mem buffer_image = (cl_mem) image.get_handle(ocl::ACCESS::RO);
        cl_mem buffer_noise = (cl_mem) noise_map.handle;
        cl_mem buffer_seg_mask = (cl_mem) seg_mask.handle;
        cl_mem buffer_bg_model = (cl_mem) bg_model.handle;
        cl_mem buffer_utility1 = (cl_mem) utility_1.handle;
        cl_mem buffer_utility2 = (cl_mem) utility_2.handle;

        auto _lbsp_kernel = (uchar) config.kernel;
        auto _lbsp_threshold = (uchar) std::min(255.f, config.lbsp_d * 255.f);
        auto _n_norm_alpha = (float) config.alpha_norm;
        auto _lbsp_0 = (ushort) denorm_lbsp_threshold(config.lbsp_0);

        auto _color_0 = (ushort) denorm_color_threshold(config.color_0);
        auto _t_lower = (ushort) config.t_lower;
        auto _t_upper = (ushort) config.t_upper;
        auto _ghost_n = (ushort) config.ghost_n;
        auto _ghost_l = (ushort) config.ghost_l;
        auto _ghost_n_inc = (ushort) config.ghost_n_inc;
        auto _ghost_n_dec = (ushort) config.ghost_n_dec;
        auto _ghost_t = (float) config.ghost_t;
        auto _d_min_alpha = (float) config.alpha_d_min;
        auto _flicker_v_inc = (float) config.v_flicker_inc;
        auto _flicker_v_dec = (float) config.v_flicker_dec;
        auto _flicker_v_cap = (float) config.v_flicker_cap;
        auto _t_scale_inc = (float) config.t_scale_inc;
        auto _t_scale_dec = (float) config.t_scale_dec;
        auto _r_scale = (float) config.r_scale;
        auto _r_cap = (float) config.r_cap;
        auto _matches_req = (uchar) config.n_matches;
        auto _model_size = (uchar) config.model_size;
        auto _channels_n = (uchar) config.color_channels;
        auto _rng_seed = (uint) time_seed();
        auto _width = (ushort) image.cols;
        auto _height = (ushort) image.rows;

        cl_uint idx_0 = 0;

        if (config.mask_xc) {
            const auto ex_mask = exclusion_p.getImage2D();

            cl_mem buffer_ex = (cl_mem) ex_mask.get_handle(ocl::ACCESS::RO);
            auto _exclusion = (uchar) (ex_mask.empty() ? 0 : 255);

            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_ex);
            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_exclusion);
        }

        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_image);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_noise);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_bg_model);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_utility1);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_utility2);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_seg_mask);

        if (config.lbsp_on) {
            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_lbsp_kernel);
            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_lbsp_threshold);
            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_n_norm_alpha);
            idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_lbsp_0);
        }

        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_color_0);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_t_lower);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_t_upper);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_n);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_l);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_n_inc);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_n_dec);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_ghost_t);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_d_min_alpha);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_flicker_v_inc);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_flicker_v_dec);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_flicker_v_cap);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_t_scale_inc);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_t_scale_dec);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_r_scale);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_r_cap);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_matches_req);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_model_size);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_channels_n);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uint), &_rng_seed);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_width);
        xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_height);

        xm::ocl::enqueue_kernel_fast(
            queue,
            kernel_subsense,
            2,
            g_size,
            l_size,
            false);


        // ============================================= MORPHOLOGY =============================================

        if (config.morph_on) {
            gate(queue, l_size, g_size);
            dilate(queue, l_size, g_size);
            erode(queue, l_size, g_size);
        }

        // ============================================= MASK APPLY ==============================================
        const auto img_out = xm::ocl::Image2D::allocate_like(original);

        cl_mem buffer_in = (cl_mem) seg_mask.handle;
        cl_mem buffer_out = (cl_mem) img_out.handle;
        cl_mem buffer_original = (cl_mem) original.get_handle(ocl::ACCESS::RO);

        auto _mask_w = (ushort) image.cols;
        auto _mask_h = (ushort) image.rows;
        auto _out_w = (ushort) img_out.cols;
        auto _out_h = (ushort) img_out.rows;
        auto _scale_w = (float) img_out.cols / (float) image.cols;
        auto _scale_h = (float) img_out.rows / (float) image.rows;
        auto _d_x = (uchar) std::ceil(_scale_w);
        auto _d_y = (uchar) std::ceil(_scale_h);
        auto _color_b = (uchar) config.color.b;
        auto _color_g = (uchar) config.color.g;
        auto _color_r = (uchar) config.color.r;

        cl_uint idx_1 = 0;
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(cl_mem), &buffer_in);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(cl_mem), &buffer_original);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(cl_mem), &buffer_out);

        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(ushort), &_mask_w);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(ushort), &_mask_h);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(ushort), &_out_w);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(ushort), &_out_h);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(float), &_scale_w);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(float), &_scale_h);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_d_x);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_d_y);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_color_b);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_color_g);
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_color_r);
        xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(uchar), &_channels_n);

        xm::ocl::enqueue_kernel_fast(
            queue,
            kernel_apply,
            2,
            g_size,
            l_size,
            false);

        return xm::ocl::iop::ClImagePromise(img_out,queue)
        .withCleanup(downscaled_p)
        .withCleanup(exclusion_p)
        .withCleanup(original_p);
    }


    void BgSubtract::gate(cl_command_queue queue, size_t *l_size, size_t *g_size) {
        if (config.refine_gate <= 0 || !config.morph_on)
            return;

        auto gate_kernel_type = (uchar) config.gate_kernel;
        auto gate_threshold = (uchar) ((float) gate_kernel_type * 4.f * config.refine_gate_threshold);
        auto gate_c_size = (uchar) 1;
        auto _width = (ushort) seg_mask.cols;
        auto _height = (ushort) seg_mask.rows;

        xm::ocl::Image2D im_1 = seg_mask;
        xm::ocl::Image2D im_2 = tmp_mask;

        for (int i = 0; i < config.refine_gate; i++) {

            cl_uint idx_2 = 0;

            cl_mem b1 = im_1.handle;
            cl_mem b2 = im_2.handle;

            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(cl_mem), &b1);
            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(cl_mem), &b2);

            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(uchar), &gate_kernel_type);
            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(uchar), &gate_c_size);
            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(uchar), &gate_threshold);
            idx_2 = xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(ushort), &_width);
            xm::ocl::set_kernel_arg(kernel_gate, idx_2, sizeof(ushort), &_height);

            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_gate,
                    2,
                    g_size,
                    l_size,
                    false);

            auto tmp = im_1;
            im_1 = std::move(im_2);
            im_2 = std::move(tmp);
        }

        seg_mask = std::move(im_1);
    }

    void BgSubtract::erode(cl_command_queue queue, size_t *l_size, size_t *g_size) {
        if (config.refine_erode <= 0 || !config.morph_on)
            return;

        auto erode_kernel_type = (uchar) config.erode_kernel;
        auto erode_c_size = (uchar) 1;
        auto _width = (ushort) seg_mask.cols;
        auto _height = (ushort) seg_mask.rows;

        xm::ocl::Image2D im_1 = seg_mask;
        xm::ocl::Image2D im_2 = tmp_mask;

        for (int i = 0; i < config.refine_erode; i++) {

            cl_uint idx_2 = 0;

            cl_mem b1 = im_1.handle;
            cl_mem b2 = im_2.handle;

            idx_2 = xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(cl_mem), &b1);
            idx_2 = xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(cl_mem), &b2);

            idx_2 = xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(uchar), &erode_kernel_type);
            idx_2 = xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(uchar), &erode_c_size);
            idx_2 = xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(ushort), &_width);
            xm::ocl::set_kernel_arg(kernel_erode, idx_2, sizeof(ushort), &_height);

            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_erode,
                    2,
                    g_size,
                    l_size,
                    false);

            auto tmp = im_1;
            im_1 = std::move(im_2);
            im_2 = std::move(tmp);
        }

        seg_mask = std::move(im_1);
    }

    void BgSubtract::dilate(cl_command_queue queue, size_t *l_size, size_t *g_size) {
        if (config.refine_dilate <= 0 || !config.morph_on)
            return;

        auto dilate_kernel_type = (uchar) config.dilate_kernel;
        auto dilate_c_size = (uchar) 1;
        auto _width = (ushort) seg_mask.cols;
        auto _height = (ushort) seg_mask.rows;

        xm::ocl::Image2D im_1 = seg_mask;
        xm::ocl::Image2D im_2 = tmp_mask;

        for (int i = 0; i < config.refine_dilate; i++) {
            cl_uint idx_3 = 0;

            cl_mem b1 = im_1.handle;
            cl_mem b2 = im_2.handle;

            idx_3 = xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(cl_mem), &b1);
            idx_3 = xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(cl_mem), &b2);
            idx_3 = xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(uchar), &dilate_kernel_type);
            idx_3 = xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(uchar), &dilate_c_size);
            idx_3 = xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(ushort), &_width);
            xm::ocl::set_kernel_arg(kernel_dilate, idx_3, sizeof(ushort), &_height);

            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_dilate,
                    2,
                    g_size,
                    l_size,
                    false);

            auto tmp = im_1;
            im_1 = std::move(im_2);
            im_2 = std::move(tmp);
        }

        seg_mask = std::move(im_1);
    }

    xm::ocl::iop::ClImagePromise BgSubtract::debug(int n, const xm::ocl::iop::ClImagePromise &ref) {
        if (!config.debug_on)
            throw std::invalid_argument("Debug is disabled");

        auto queue = ref.queue() == nullptr ? retrieve_queue(-1) : ref.queue();

        auto out = xm::ocl::Image2D::allocate(
                utility_1.cols, utility_1.rows, (size_t) 3, 1,
                ocl_context, device_id);

        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) out.cols, pref_size),
                            xm::ocl::optimal_global_size((int) out.rows, pref_size)};

        cl_mem buffer_bg_model = (cl_mem) bg_model.handle;
        cl_mem buffer_seg_mask = (cl_mem) seg_mask.handle;
        cl_mem buffer_utility1 = (cl_mem) utility_1.handle;
        cl_mem buffer_utility2 = (cl_mem) utility_2.handle;
        cl_mem buffer_noise = (cl_mem) noise_map.handle;
        cl_mem buffer_out = (cl_mem) out.handle;

        auto _lbsp_kernel = (uchar) config.kernel;
        auto _model_size = (uchar) config.model_size;
        auto _select_n = (uchar) n;
        auto _flicker_v_cap = (float) config.v_flicker_cap;
        auto _r_cap = (float) config.r_cap;
        auto _rng_seed = (uint) time_seed();
        auto _ghost_n = (ushort) config.ghost_n;
        auto _width = (ushort) out.cols;
        auto _height = (ushort) out.rows;

        cl_uint idx_1 = 0;
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_bg_model);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_seg_mask);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_utility1);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_utility2);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_noise);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(cl_mem), &buffer_out);

        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(uchar), &_lbsp_kernel);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(uchar), &_model_size);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(uchar), &_select_n);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(float), &_flicker_v_cap);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(float), &_r_cap);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(uint), &_rng_seed);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(ushort), &_ghost_n);
        idx_1 = xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(ushort), &_width);
        xm::ocl::set_kernel_arg(kernel_debug, idx_1, sizeof(ushort), &_height);

        xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_debug,
                2,
                g_size,
                l_size,
                false);

        return xm::ocl::iop::ClImagePromise(out, queue)
        .withCleanup(ref);
    }

    int BgSubtract::denorm_color_threshold(float v) const {
        return (int) std::ceil(v * (config.norm_l2
            ? std::sqrt((float) config.color_channels * std::pow(255.f, 2.f))
            : (float) config.color_channels * 255
        ));
    }

    int BgSubtract::denorm_lbsp_threshold(float v) const {
        return (int) std::ceil((float) config.color_channels * v * 4.f * (float) ((int) config.kernel));
    }

    void new_size(const int w, const int h, const int base, int &new_w, int &new_h, float &scale) {
        const auto ratio = (float) w / (float) h;
        if (w < h) {
            new_w = base;
            new_h = (int) ((float) base / ratio);
            scale = (float) w / (float) base;
        } else {
            new_h = base;
            new_w = (int) ((float) base * ratio);
            scale = (float) h / (float) base;
        }
    }

    uint32_t time_seed() {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto duration = now.time_since_epoch();
        const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        return static_cast<uint32_t>(nanoseconds & 0xFFFFFFFF);
    }

    int bgs::lbsp_k_size_bytes(bgs::KernelType t) {
        if (t == bgs::KERNEL_TYPE_NONE) return 0;
        return ((int) t) < ((int) bgs::KERNEL_TYPE_RUBY_12) ? 1 : 2;
    }

    void BgSubtract::start() {
        ready = true;
    }

    void BgSubtract::stop() {
        ready = false;
    }

    void BgSubtract::reset() {
        model_i = 0;
    }

    void BgSubtract::set_debug_mode(int mode) {
        debug_mode = mode;
    }

}
#pragma clang diagnostic pop