//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/filter/bg_lbp_subtract.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../kernels/subsense.h"

namespace xm::filters {

    BgLbpSubtract::BgLbpSubtract() {
        device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        ocl_command_queue = xm::ocl::create_queue_device(
                ocl_context,
                device_id,
                true,
                false);

        program_subsense = xm::ocl::build_program(
            ocl_context, device_id,
            ocl_kernel_subsense_data,
            ocl_kernel_subsense_data_size,
            "subsense.cl",
            "-DDISABLED_EXCLUSION_MASK"
            );

        kernel_apply = xm::ocl::build_kernel(program_subsense, "kernel_upscale_apply");
        kernel_prepare = xm::ocl::build_kernel(program_subsense, "kernel_prepare_model");
        kernel_subsense = xm::ocl::build_kernel(program_subsense, "kernel_subsense");
        kernel_downscale = xm::ocl::build_kernel(program_subsense, "kernel_downscale");
        kernel_upscale = xm::ocl::build_kernel(program_subsense, "kernel_upscale");
        kernel_dilate = xm::ocl::build_kernel(program_subsense, "kernel_dilate");
        kernel_erode = xm::ocl::build_kernel(program_subsense, "kernel_erode");

        pref_size = xm::ocl::optimal_local_size(device_id, kernel_subsense);
    }

    BgLbpSubtract::~BgLbpSubtract() {
        for (auto &item: ocl_queue_map) {
            if (item.second == nullptr)
                continue;
            clReleaseCommandQueue(item.second);
        }

        clReleaseKernel(kernel_apply);
        clReleaseKernel(kernel_prepare);
        clReleaseKernel(kernel_subsense);
        clReleaseKernel(kernel_downscale);
        clReleaseKernel(kernel_upscale);
        clReleaseKernel(kernel_dilate);
        clReleaseKernel(kernel_erode);
        clReleaseProgram(program_subsense);
        clReleaseCommandQueue(ocl_command_queue);
        clReleaseContext(ocl_context);
        clReleaseDevice(device_id);
    }

    void BgLbpSubtract::start() {
        ready = true;
    }

    void BgLbpSubtract::stop() {
        ready = false;
    }

    void BgLbpSubtract::reset() {
        model_i = 0;
    }

    cl_command_queue BgLbpSubtract::retrieve_queue(int index) {
        if (index <= 0)
            return ocl_command_queue;

        if (ocl_queue_map.contains(index))
            return ocl_queue_map[index];

        ocl_queue_map.emplace(index, xm::ocl::create_queue_device(
                ocl_context,
                device_id,
                true,
                false));
        return ocl_queue_map[index];
    }

    void BgLbpSubtract::init(const bgs::Conf &conf) {
//        kernel_type = conf.kernel_type;
//        bgr_bg_color = conf.color;
        initialized = true;
        reset();
    }

    xm::ocl::iop::ClImagePromise BgLbpSubtract::filter(const ocl::iop::ClImagePromise &frame_in, int q_idx) {
        if (!initialized)
            throw std::logic_error("Filter is not initialized");

        if (!ready)
            return frame_in;

        auto downscaled = downscale(frame_in, BASE_RESOLUTION, q_idx);

        if (model_i < model_size) {
            prepare_update_model(downscaled, q_idx);
            model_i += 1;
            return downscaled;
        }

        return subsense(downscaled, frame_in, {/*TODO*/}, q_idx);
    }

    void BgLbpSubtract::prepare_update_model(const ocl::iop::ClImagePromise &in_p, int q_idx) {
        cl_command_queue queue = q_idx < 0 && in_p.queue() != nullptr ? in_p.queue() : retrieve_queue(q_idx);
        const auto &in = in_p.getImage2D();

        const int n_w = in.cols;
        const int n_h = in.rows;

        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) n_w, pref_size),
                            xm::ocl::optimal_global_size((int) n_h, pref_size)};

        const int lbsp_c_size = bgs::lbsp_k_size_bytes(kernel_type);
        if (bg_model.empty()) {
            bg_model = xm::ocl::Image2D::allocate(
                n_w, n_h, model_size * (color_c + color_c * lbsp_c_size), 1,
                ocl_context, device_id);
        }

        if (utility_1.empty()) {
            utility_1 = xm::ocl::Image2D::allocate(
                n_w, n_h, 4, sizeof(float),
                ocl_context, device_id);
        }

        if (utility_2.empty()) {
            utility_2 = xm::ocl::Image2D::allocate(
                n_w, n_h, 3, sizeof(short),
                ocl_context, device_id);
        }

        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.get_handle(ocl::ACCESS::RO);
        cl_mem buffer_bg_model = (cl_mem) bg_model.handle;
        cl_mem buffer_utility1 = (cl_mem) utility_1.handle;
        cl_mem buffer_utility2 = (cl_mem) utility_2.handle;

        auto lbsp_kernel = (uchar) kernel_type;
        auto lbsp_threshold = (uchar) std::min(255.f, threshold_lbsp * 255.f);
        auto _model_i = (uchar) model_i;
        auto _model_size = (uchar) model_size;
        auto _channels_n = (uchar) color_c;
        auto _t_lower = (ushort) t_lower;
        auto _width = (ushort) n_w;
        auto _height = (ushort) n_h;

        int idx_1 = 0;
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_in);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_bg_model);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_utility1);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_utility2);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &lbsp_kernel);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &lbsp_threshold);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_model_i);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_model_size);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(uchar), &_channels_n);
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(ushort), &_t_lower);
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

    xm::ocl::iop::ClImagePromise BgLbpSubtract::downscale(const ocl::iop::ClImagePromise &in_p, int base, int q_idx) {
        cl_command_queue queue = q_idx < 0 && in_p.queue() != nullptr ? in_p.queue() : retrieve_queue(q_idx);
        const auto &in = in_p.getImage2D();

        float scale;
        int n_w, n_h;

        new_size(in.cols, in.rows, base, n_w, n_h, scale);
        const int inter_size = n_w * n_h * color_c * sizeof(char);
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
        auto channels_n = (uchar) color_c;
        auto is_linear = (uchar) linear ? 255 : 0;

        int idx_0 = 0;
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
                n_w, n_h, color_c, sizeof(uchar),
                buffer_io_1, ocl_context, device_id),
                queue);
    }

    xm::ocl::iop::ClImagePromise BgLbpSubtract::subsense(const ocl::iop::ClImagePromise &downscaled_p,
                                                         const ocl::iop::ClImagePromise &original_p,
                                                         const ocl::iop::ClImagePromise &exclusion_p,
                                                         int q_idx) {
        cl_command_queue queue = q_idx < 0 && downscaled_p.queue() != nullptr
            ? downscaled_p.queue()
            : retrieve_queue(q_idx);

        const auto image = downscaled_p.getImage2D();
        const auto original = original_p.getImage2D();

        const int inter_size = image.cols * image.rows * 1 * sizeof(char);
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) image.cols, pref_size),
                            xm::ocl::optimal_global_size((int) image.rows, pref_size)};

        cl_int err;

        cl_mem buffer_image = (cl_mem) image.get_handle(ocl::ACCESS::RO);
        cl_mem buffer_bg_model = (cl_mem) bg_model.get_handle(ocl::ACCESS::RW);
        cl_mem buffer_utility1 = (cl_mem) utility_1.get_handle(ocl::ACCESS::RW);
        cl_mem buffer_utility2 = (cl_mem) utility_2.get_handle(ocl::ACCESS::RW);
        cl_mem buffer_seg_mask = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, inter_size, NULL, &err);

        auto _lbsp_kernel = (uchar) kernel_type;
        auto _lbsp_threshold = (uchar) std::min(255.f, threshold_lbsp * 255.f);
        auto _n_norm_alpha = (float) alpha_norm;
        auto _lbsp_0 = (ushort) lbsp_0;

        auto _color_0 = (ushort) color_0;
        auto _t_lower = (ushort) t_lower;
        auto _t_upper = (ushort) t_upper;
        auto _ghost_n = (ushort) ghost_n;
        auto _ghost_l = (ushort) ghost_l;
        auto _ghost_t = (float) ghost_t;
        auto _d_min_alpha = (float) alpha_d_min;
        auto _flicker_v_inc = (float) v_flicker_inc;
        auto _flicker_v_dec = (float) v_flicker_dec;
        auto _t_scale_inc = (float) t_scale_inc;
        auto _t_scale_dec = (float) t_scale_dec;
        auto _r_scale = (float) scale_r;
        auto _matches_req = (uchar) n_matches;
        auto _model_size = (uchar) model_size;
        auto _channels_n = (uchar) color_c;
        auto _rng_seed = (uint) time_seed();
        auto _width = (ushort) image.cols;
        auto _height = (ushort) image.rows;

        int idx_0 = 0;
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_image);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_bg_model);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_utility1);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_utility2);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(cl_mem), &buffer_seg_mask);

        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_lbsp_kernel);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(uchar), &_lbsp_threshold);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_n_norm_alpha);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_lbsp_0);

        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_color_0);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_t_lower);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_t_upper);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_n);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(ushort), &_ghost_l);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_ghost_t);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_d_min_alpha);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_flicker_v_inc);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_flicker_v_dec);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_t_scale_inc);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_t_scale_dec);
        idx_0 = xm::ocl::set_kernel_arg(kernel_subsense, idx_0, sizeof(float), &_r_scale);
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

        // TODO MASK REFINEMENT

        const auto img_out = xm::ocl::Image2D::allocate_like(original);

        cl_mem buffer_out = (cl_mem) img_out.get_handle(ocl::ACCESS::WO);
        cl_mem buffer_original = (cl_mem) original.get_handle(ocl::ACCESS::RO);

        auto _mask_w = (ushort) image.cols;
        auto _mask_h = (ushort) image.rows;
        auto _out_w = (ushort) img_out.cols;
        auto _out_h = (ushort) img_out.rows;
        auto _scale_w = (float) img_out.cols / (float) image.cols;
        auto _scale_h = (float) img_out.rows / (float) image.rows;
        auto _d_x = (uchar) std::ceil(_scale_w);
        auto _d_y = (uchar) std::ceil(_scale_h);
        auto _color_b = (uchar) bgr_bg_color.b;
        auto _color_g = (uchar) bgr_bg_color.g;
        auto _color_r = (uchar) bgr_bg_color.r;

        int idx_1 = 0;
        idx_1 = xm::ocl::set_kernel_arg(kernel_apply, idx_1, sizeof(cl_mem), &buffer_seg_mask);
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
        .withCleanup(new std::function<void()>([buffer_seg_mask]() {
            clReleaseMemObject(buffer_seg_mask);
        }));
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
        if (t == KernelType::KERNEL_TYPE_NONE)
            return 0;
        if (t == KernelType::KERNEL_TYPE_DIAMOND_16)
            return 2;
        return 1;
    }
}