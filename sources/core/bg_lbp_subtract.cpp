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

        if (model_i < model_size) {
            prepare_update_model(frame_in, q_idx);
            model_i += 1;
            return frame_in;
        }

        return frame_in;
    }

    void BgLbpSubtract::prepare_update_model(const ocl::iop::ClImagePromise &in_p, int q_idx) {
        cl_command_queue queue = q_idx < 0 && in_p.queue() != nullptr ? in_p.queue() : retrieve_queue(q_idx);
        const auto &in = in_p.getImage2D();

        float scale;
        int n_w, n_h;
        new_size(in.cols, in.rows, 240, n_w, n_h, scale);

        const int inter_size = n_w * n_h * color_c * sizeof(char);

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

        cl_int err;

        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.handle;
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
        idx_1 = xm::ocl::set_kernel_arg(kernel_prepare, idx_1, sizeof(cl_mem), &buffer_io_1);
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
            kernel_downscale,
            2,
            g_size,
            l_size,
            false);

        xm::ocl::enqueue_kernel_fast(
            queue,
            kernel_prepare,
            2,
            g_size,
            l_size,
            false);

        clReleaseMemObject(buffer_io_1);
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

    int bgs::lbsp_k_size_bytes(bgs::KernelType t) {
        if (t == KernelType::KERNEL_TYPE_NONE)
            return 0;
        if (t == KernelType::KERNEL_TYPE_DIAMOND_16)
            return 2;
        return 1;
    }
}