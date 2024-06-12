//
// Created by henryco on 6/3/24.
//

#include <iostream>
#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"
#pragma ide diagnostic ignored "UnusedLocalVariable"
#pragma ide diagnostic ignored "bugprone-easily-swappable-parameters"

#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/ocl/ocl_kernels.h"
#include "../../xmotion/core/ocl/ocl_kernels_chromakey.h"
#include "../../xmotion/core/ocl/ocl_interop.h"
#include <CL/cl.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <thread>

namespace xm::ocl {

    namespace aux {
        std::mutex GLOBAL_MUTEX;
    }

    Kernels::Kernels() {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        thread_id = oss.str();

        std::lock_guard<std::mutex> lock(aux::GLOBAL_MUTEX);

        log->debug("[{}] oCL kernels initialization, profiling: {}", thread_id, aux::DEBUG);

//        device_id = xm::ocl::find_gpu_device();
//        ocl_context = xm::ocl::create_context(device_id);
//        svm_supported = check_svm_cap(device_id);

        device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        svm_supported = check_svm_cap(device_id);

        ocl_command_queue = xm::ocl::create_queue_device(ocl_context, device_id, true, aux::DEBUG);

        program_blur = xm::ocl::build_program(ocl_context, device_id, kernels::GAUSSIAN_BLUR_KERNEL);
        kernel_blur_h = xm::ocl::build_kernel(program_blur, "gaussian_blur_horizontal");
        kernel_blur_v = xm::ocl::build_kernel(program_blur, "gaussian_blur_vertical");
        blur_local_size = xm::ocl::optimal_local_size(device_id, kernel_blur_h);

        program_dilate = xm::ocl::build_program(ocl_context, device_id, kernels::DILATE_GRAY_KERNEL);
        kernel_dilate_h = xm::ocl::build_kernel(program_dilate, "dilate_horizontal");
        kernel_dilate_v = xm::ocl::build_kernel(program_dilate, "dilate_vertical");
        dilate_local_size = xm::ocl::optimal_local_size(device_id, kernel_dilate_h);

        program_erode = xm::ocl::build_program(ocl_context, device_id, kernels::ERODE_GRAY_KERNEL);
        kernel_erode_h = xm::ocl::build_kernel(program_erode, "erode_horizontal");
        kernel_erode_v = xm::ocl::build_kernel(program_erode, "erode_vertical");
        erode_local_size = xm::ocl::optimal_local_size(device_id, kernel_erode_h);

        program_range_hls = xm::ocl::build_program(ocl_context, device_id, kernels::BGR_HLS_RANGE_KERNEL);
        kernel_range_hls = xm::ocl::build_kernel(program_range_hls, "in_range_hls");
        range_hls_local_size = xm::ocl::optimal_local_size(device_id, kernel_range_hls);

        program_mask_apply = xm::ocl::build_program(ocl_context, device_id, kernels::MASK_APPLY_BG_FG);
        kernel_mask_apply = xm::ocl::build_kernel(program_mask_apply, "apply_mask");
        mask_apply_local_size = xm::ocl::optimal_local_size(device_id, kernel_mask_apply);

        program_power_chroma = xm::ocl::build_program(ocl_context, device_id, kernels::CHROMA_KEY_COMPLEX);
        kernel_power_chroma = xm::ocl::build_kernel(program_power_chroma, "power_chromakey");
        kernel_power_apply = xm::ocl::build_kernel(program_power_chroma, "power_apply");
        kernel_power_mask = xm::ocl::build_kernel(program_power_chroma, "power_mask");
        power_chroma_local_size = xm::ocl::optimal_local_size(device_id, kernel_power_chroma);

        for (int i = 1; i < ((31 - 1) / 2); i++) {
            cv::UMat kernel_mat;
            const auto k_size = (i * 2) + 1;
            cv::getGaussianKernel(k_size, ((float) k_size - 1.f) / 6.f, CV_32F).copyTo(kernel_mat);
            blur_kernels[i] = xm::ocl::iop::from_cv_umat(kernel_mat, ocl_context, device_id, xm::ocl::ACCESS::RO);
        }

        log->debug("[{}] oCL kernels initialized", thread_id);
    }

    Kernels::~Kernels() {
        log->debug("[{}] releasing oCL kernels", thread_id);

        clReleaseKernel(kernel_blur_h);
        clReleaseKernel(kernel_blur_v);
        clReleaseProgram(program_blur);

        clReleaseKernel(kernel_dilate_h);
        clReleaseKernel(kernel_dilate_v);
        clReleaseProgram(program_dilate);

        clReleaseKernel(kernel_erode_h);
        clReleaseKernel(kernel_erode_v);
        clReleaseProgram(program_erode);

        clReleaseKernel(kernel_range_hls);
        clReleaseProgram(program_range_hls);

        clReleaseKernel(kernel_mask_apply);
        clReleaseProgram(program_mask_apply);

        for (auto &item: ocl_queue_map) {
            if (item.second == nullptr)
                continue;
            clReleaseCommandQueue(item.second);
        }

        clReleaseCommandQueue(ocl_command_queue);
        clReleaseContext(ocl_context);
        clReleaseDevice(device_id);

        log->debug("[{}] released oCL kernels", thread_id);
    }

    void Kernels::print_time(cl_ulong time, const std::string &name, bool force) const {
        if (aux::DEBUG || force)
            log->debug("[{}] kernel [{}] execution time: {} ns", thread_id, name, std::to_string(time));
    }

    cl_command_queue Kernels::retrieve_queue(int index) {
        if (index <= 0)
            return ocl_command_queue;

        if (ocl_queue_map.contains(index))
            return ocl_queue_map[index];

        ocl_queue_map.emplace(index, xm::ocl::create_queue_device(
                ocl_context,
                device_id,
                true,
                aux::DEBUG));
        return ocl_queue_map[index];
    }

    void blur(const cv::UMat &in, cv::UMat &out, const int kernel_size, int queue_index) {
        if (kernel_size < 3 || kernel_size % 2 == 0 || kernel_size > 31)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));
        cv::UMat result_1(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kernel_mat_buffer = (cl_mem) Kernels::instance().blur_kernels[(kernel_size - 1) / 2].handle;
        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto pref_size = Kernels::instance().blur_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(in.cols, pref_size),
                            xm::ocl::optimal_global_size(in.rows, pref_size)};

        {
            auto kernel = Kernels::instance().kernel_blur_h;
            auto input_buffer = (cl_mem) in.handle(cv::ACCESS_READ);
            auto result_1_buffer = (cl_mem) result_1.handle(cv::ACCESS_WRITE);

            auto width = (uint) in.cols;
            auto height = (uint) in.rows;

            xm::ocl::set_kernel_arg(kernel, 0, sizeof(cl_mem), &input_buffer);
            xm::ocl::set_kernel_arg(kernel, 1, sizeof(cl_mem), &kernel_mat_buffer);
            xm::ocl::set_kernel_arg(kernel, 2, sizeof(cl_mem), &result_1_buffer);
            xm::ocl::set_kernel_arg(kernel, 3, sizeof(uint), &width);
            xm::ocl::set_kernel_arg(kernel, 4, sizeof(uint), &height);
            xm::ocl::set_kernel_arg(kernel, 5, sizeof(int), &kh_size);

            const auto time = xm::ocl::enqueue_kernel_sync(
                    queue, kernel, 2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            Kernels::instance().print_time(time, "blur_h");
        }

        {
            auto kernel = Kernels::instance().kernel_blur_v;
            auto result_1_buffer = (cl_mem) result_1.handle(cv::ACCESS_READ);
            auto result_2_buffer = (cl_mem) result_2.handle(cv::ACCESS_WRITE);
            auto width = (uint) in.cols;
            auto height = (uint) in.rows;

            xm::ocl::set_kernel_arg(kernel, 0, sizeof(cl_mem), &result_1_buffer);
            xm::ocl::set_kernel_arg(kernel, 1, sizeof(cl_mem), &kernel_mat_buffer);
            xm::ocl::set_kernel_arg(kernel, 2, sizeof(cl_mem), &result_2_buffer);
            xm::ocl::set_kernel_arg(kernel, 3, sizeof(uint), &width);
            xm::ocl::set_kernel_arg(kernel, 4, sizeof(uint), &height);
            xm::ocl::set_kernel_arg(kernel, 5, sizeof(int), &kh_size);

            const auto time = xm::ocl::enqueue_kernel_sync(
                    queue, kernel, 2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            Kernels::instance().print_time(time, "blur_v");
        }

        out = std::move(result_2);
    }

    void bgr_in_range_hls(const cv::Scalar &hls_low, const cv::Scalar &hls_up, const cv::UMat &in, cv::UMat &out, int queue_index) {
        cv::UMat result(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto pref_size = Kernels::instance().range_hls_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(in.cols, pref_size),
                            xm::ocl::optimal_global_size(in.rows, pref_size)};

        {
            auto kernel = Kernels::instance().kernel_range_hls;
            auto buffer_in = (cl_mem) in.handle(cv::ACCESS_READ);
            auto buffer_out = (cl_mem) result.handle(cv::ACCESS_WRITE);
            auto width = (uint) in.cols;
            auto height = (uint) in.rows;
            auto lower_h = (uchar) hls_low[0];
            auto lower_l = (uchar) hls_low[1];
            auto lower_s = (uchar) hls_low[2];
            auto upper_h = (uchar) hls_up[0];
            auto upper_l = (uchar) hls_up[1];
            auto upper_s = (uchar) hls_up[2];

            xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_out);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(uint), &width);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &height);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uchar), &lower_h);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 5, sizeof(uchar), &lower_l);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 6, sizeof(uchar), &lower_s);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 7, sizeof(uchar), &upper_h);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 8, sizeof(uchar), &upper_l);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 9, sizeof(uchar), &upper_s);

            const auto time = xm::ocl::enqueue_kernel_sync(
                    queue, kernel, 2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            Kernels::instance().print_time(time, "range_hls");
        }

        out = std::move(result);
    }

    void dilate(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size, int queue_index) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto pref_size = Kernels::instance().dilate_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(in.cols, pref_size),
                            xm::ocl::optimal_global_size(in.rows, pref_size)};

        for (int i = 0; i < iterations; i++) {
            {
                auto kernel = Kernels::instance().kernel_dilate_h;
                auto buffer_in = i > 0
                                 ? (cl_mem) result_2.handle(cv::ACCESS_READ)
                                 : (cl_mem) in.handle(cv::ACCESS_READ);
                auto buffer_out = (cl_mem) result_1.handle(cv::ACCESS_WRITE);
                auto width = (uint) in.cols;
                auto height = (uint) in.rows;

                xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_out);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(int), &kh_size);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &width);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &height);

                const auto time = xm::ocl::enqueue_kernel_sync(
                        queue, kernel, 2,
                        g_size,
                        l_size,
                        aux::DEBUG);
                Kernels::instance().print_time(time, "dilate_h");
            }

            {
                auto kernel = Kernels::instance().kernel_dilate_v;
                auto buffer_in = (cl_mem) result_1.handle(cv::ACCESS_READ);
                auto buffer_out = (cl_mem) result_2.handle(cv::ACCESS_WRITE);
                auto width = (uint) in.cols;
                auto height = (uint) in.rows;

                xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_out);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(int), &kh_size);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &width);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &height);

                const auto time = xm::ocl::enqueue_kernel_sync(
                        queue, kernel, 2,
                        g_size,
                        l_size,
                        aux::DEBUG);
                Kernels::instance().print_time(time, "dilate_v");
            }
        }

        out = std::move(result_2);
    }

    void erode(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size, int queue_index) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto pref_size = Kernels::instance().erode_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(in.cols, pref_size),
                            xm::ocl::optimal_global_size(in.rows, pref_size)};

        for (int i = 0; i < iterations; i++) {
            {
                auto kernel = Kernels::instance().kernel_erode_h;
                auto buffer_in = i > 0
                                 ? (cl_mem) result_2.handle(cv::ACCESS_READ)
                                 : (cl_mem) in.handle(cv::ACCESS_READ);
                auto buffer_out = (cl_mem) result_1.handle(cv::ACCESS_WRITE);
                auto width = (uint) in.cols;
                auto height = (uint) in.rows;

                xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_out);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(int), &kh_size);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &width);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &height);

                const auto time = xm::ocl::enqueue_kernel_sync(
                        queue, kernel, 2,
                        g_size,
                        l_size,
                        aux::DEBUG);
                Kernels::instance().print_time(time, "erode_h");
            }

            {
                auto kernel = Kernels::instance().kernel_erode_v;
                auto buffer_in = (cl_mem) result_1.handle(cv::ACCESS_READ);
                auto buffer_out = (cl_mem) result_2.handle(cv::ACCESS_WRITE);
                auto width = (uint) in.cols;
                auto height = (uint) in.rows;

                xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_out);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(int), &kh_size);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &width);
                xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &height);

                const auto time = xm::ocl::enqueue_kernel_sync(
                        queue, kernel, 2,
                        g_size,
                        l_size,
                        aux::DEBUG);
                Kernels::instance().print_time(time, "erode_v");
            }
        }

        out = std::move(result_2);
    }

    void apply_mask_with_color(const cv::Scalar &color, const cv::UMat &img, const cv::UMat &mask, cv::UMat &out, int queue_index) {
        cv::UMat result(img.rows, img.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto pref_size = Kernels::instance().mask_apply_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(img.cols, pref_size),
                            xm::ocl::optimal_global_size(img.rows, pref_size)};

        auto kernel = Kernels::instance().kernel_mask_apply;

        auto buffer_mask = (cl_mem) mask.handle(cv::ACCESS_READ);
        auto buffer_image = (cl_mem) img.handle(cv::ACCESS_READ);
        auto buffer_result = (cl_mem) result.handle(cv::ACCESS_WRITE);
        auto mask_width = (uint) mask.cols;
        auto mask_height = (uint) mask.rows;
        auto width = (uint) img.cols;
        auto height = (uint) img.rows;
        auto s_w = (float) mask.cols / (float) img.cols;
        auto s_h = (float) mask.rows / (float) img.rows;
        auto b = (uchar) color[0];
        auto g = (uchar) color[1];
        auto r = (uchar) color[2];

        xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_mask);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_image);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(cl_mem), &buffer_result);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(uint), &mask_width);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &mask_height);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 5, sizeof(uint), &width);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 6, sizeof(uint), &height);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 7, sizeof(cl_float), &s_w);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 8, sizeof(cl_float), &s_h);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 9, sizeof(uchar), &b);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 10, sizeof(uchar), &g);
        xm::ocl::set_kernel_arg(kernel, (cl_uint) 11, sizeof(uchar), &r);

        const auto time = xm::ocl::enqueue_kernel_sync(
                queue, kernel, 2,
                g_size,
                l_size,
                aux::DEBUG);
        Kernels::instance().print_time(time, "apply_mask");

        out = result;
    }

    void chroma_key(const cv::UMat &in, cv::UMat &out,
                    const cv::Scalar &hls_low,
                    const cv::Scalar &hls_up,
                    const cv::Scalar &color,
                    bool linear,
                    int mask_size,
                    int blur,
                    int fine,
                    int refine,
                    int queue_index) {
        cv::UMat result(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (int) ((float) n_w / ratio);

        // power_mask -> (erode_h -> erode_v) -> (dilate_h -> dilate_v) -> mask_apply

        cl_int err;

        const auto context = Kernels::instance().ocl_context;
        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto inter_size = n_w * n_h * 3;
        const auto pref_size = Kernels::instance().mask_apply_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(n_w, pref_size),
                            xm::ocl::optimal_global_size(n_h, pref_size)};


        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.handle(cv::ACCESS_READ);
        cl_mem buffer_blur = (cl_mem) Kernels::instance().blur_kernels[(blur - 1) / 2].handle;
        cl_mem buffer_io_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);
        cl_mem buffer_io_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);
        cl_mem buffer_out = (cl_mem) result.handle(cv::ACCESS_WRITE);


        // ======= KERNELS ALLOCATION !
        auto kernel_power_mask = Kernels::instance().kernel_power_mask;
        auto kernel_erode_h = Kernels::instance().kernel_erode_h;
        auto kernel_erode_v = Kernels::instance().kernel_erode_v;
        auto kernel_dilate_h = Kernels::instance().kernel_dilate_h;
        auto kernel_dilate_v = Kernels::instance().kernel_dilate_v;
        auto kernel_power_apply = Kernels::instance().kernel_power_apply;


        // ======= KERNEL PARAMETERS !
        auto morph_kern_half_size = (int) (fine / 2);
        auto blur_kern_half_size = (int) (blur / 2);
        auto mask_height = (uint) n_h;
        auto mask_width = (uint) n_w;
        auto out_height = (uint) in.rows;
        auto out_width = (uint) in.cols;
        auto scale_h = (float) in.rows / (float) n_h;
        auto scale_w = (float) in.cols / (float) n_w;
        auto lower_h = (uchar) hls_low[0];
        auto lower_l = (uchar) hls_low[1];
        auto lower_s = (uchar) hls_low[2];
        auto upper_h = (uchar) hls_up[0];
        auto upper_l = (uchar) hls_up[1];
        auto upper_s = (uchar) hls_up[2];
        auto color_b = (uchar) color[0];
        auto color_g = (uchar) color[1];
        auto color_r = (uchar) color[2];
        auto is_linear = (uchar) linear ? 1 : 0;
        auto is_blur = (uchar) (blur >= 3);
        auto dx = (uint) std::ceil(scale_w);
        auto dy = (uint) std::ceil(scale_h);


        // ======= KERNEL ARGUMENTS !
        {
            xm::ocl::set_kernel_arg(kernel_power_mask, 0, sizeof(cl_mem), &buffer_in);
            xm::ocl::set_kernel_arg(kernel_power_mask, 1, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_power_mask, 2, sizeof(cl_mem), &buffer_blur);
            xm::ocl::set_kernel_arg(kernel_power_mask, 3, sizeof(int), &blur_kern_half_size);
            xm::ocl::set_kernel_arg(kernel_power_mask, 4, sizeof(uchar), &is_blur);
            xm::ocl::set_kernel_arg(kernel_power_mask, 5, sizeof(uchar), &is_linear);
            xm::ocl::set_kernel_arg(kernel_power_mask, 6, sizeof(uint), &out_width);
            xm::ocl::set_kernel_arg(kernel_power_mask, 7, sizeof(uint), &out_height);
            xm::ocl::set_kernel_arg(kernel_power_mask, 8, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_power_mask, 9, sizeof(uint), &mask_height);
            xm::ocl::set_kernel_arg(kernel_power_mask, 10, sizeof(float), &scale_w);
            xm::ocl::set_kernel_arg(kernel_power_mask, 11, sizeof(float), &scale_h);
            xm::ocl::set_kernel_arg(kernel_power_mask, 12, sizeof(uchar), &lower_h);
            xm::ocl::set_kernel_arg(kernel_power_mask, 13, sizeof(uchar), &lower_l);
            xm::ocl::set_kernel_arg(kernel_power_mask, 14, sizeof(uchar), &lower_s);
            xm::ocl::set_kernel_arg(kernel_power_mask, 15, sizeof(uchar), &upper_h);
            xm::ocl::set_kernel_arg(kernel_power_mask, 16, sizeof(uchar), &upper_l);
            xm::ocl::set_kernel_arg(kernel_power_mask, 17, sizeof(uchar), &upper_s);
        }

        if (fine >= 3 && refine > 0) {
            xm::ocl::set_kernel_arg(kernel_erode_h, 0, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_erode_h, 1, sizeof(cl_mem), &buffer_io_2);
            xm::ocl::set_kernel_arg(kernel_erode_h, 2, sizeof(int), &morph_kern_half_size);
            xm::ocl::set_kernel_arg(kernel_erode_h, 3, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_erode_h, 4, sizeof(uint), &mask_height);

            xm::ocl::set_kernel_arg(kernel_erode_v, 0, sizeof(cl_mem), &buffer_io_2);
            xm::ocl::set_kernel_arg(kernel_erode_v, 1, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_erode_v, 2, sizeof(int), &morph_kern_half_size);
            xm::ocl::set_kernel_arg(kernel_erode_v, 3, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_erode_v, 4, sizeof(uint), &mask_height);

            xm::ocl::set_kernel_arg(kernel_dilate_h, 0, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_dilate_h, 1, sizeof(cl_mem), &buffer_io_2);
            xm::ocl::set_kernel_arg(kernel_dilate_h, 2, sizeof(int), &morph_kern_half_size);
            xm::ocl::set_kernel_arg(kernel_dilate_h, 3, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_dilate_h, 4, sizeof(uint), &mask_height);

            xm::ocl::set_kernel_arg(kernel_dilate_v, 0, sizeof(cl_mem), &buffer_io_2);
            xm::ocl::set_kernel_arg(kernel_dilate_v, 1, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_dilate_v, 2, sizeof(int), &morph_kern_half_size);
            xm::ocl::set_kernel_arg(kernel_dilate_v, 3, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_dilate_v, 4, sizeof(uint), &mask_height);
        }

        {
            xm::ocl::set_kernel_arg(kernel_power_apply, 0, sizeof(cl_mem), &buffer_in);
            xm::ocl::set_kernel_arg(kernel_power_apply, 1, sizeof(cl_mem), &buffer_io_1);
            xm::ocl::set_kernel_arg(kernel_power_apply, 2, sizeof(cl_mem), &buffer_out);
            xm::ocl::set_kernel_arg(kernel_power_apply, 3, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel_power_apply, 4, sizeof(uint), &mask_height);
            xm::ocl::set_kernel_arg(kernel_power_apply, 5, sizeof(uint), &out_width);
            xm::ocl::set_kernel_arg(kernel_power_apply, 6, sizeof(uint), &out_height);
            xm::ocl::set_kernel_arg(kernel_power_apply, 7, sizeof(cl_float), &scale_w);
            xm::ocl::set_kernel_arg(kernel_power_apply, 8, sizeof(cl_float), &scale_h);
            xm::ocl::set_kernel_arg(kernel_power_apply, 9, sizeof(uint), &dx);
            xm::ocl::set_kernel_arg(kernel_power_apply, 10, sizeof(uint), &dy);
            xm::ocl::set_kernel_arg(kernel_power_apply, 11, sizeof(uchar), &color_b);
            xm::ocl::set_kernel_arg(kernel_power_apply, 12, sizeof(uchar), &color_g);
            xm::ocl::set_kernel_arg(kernel_power_apply, 13, sizeof(uchar), &color_r);
        }


        // ======= KERNEL EVENTS !
        cl_event *erode_h_events = new cl_event[std::max(0, refine)];
        cl_event *erode_v_events = new cl_event[std::max(0, refine)];
        cl_event *dilate_h_events = new cl_event[std::max(0, refine)];
        cl_event *dilate_v_events = new cl_event[std::max(0, refine)];
        cl_event maks_range_event;
        cl_event maks_apply_event;

        xm::ocl::finish_queue(queue);

        // ======= KERNEL ENQUEUE !
        maks_range_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_power_mask,
                2,
                g_size,
                l_size,
                aux::DEBUG);

        for (int i = 0; i < refine; i++) {
            erode_h_events[i] = xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_erode_h,
                    2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            erode_v_events[i] = xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_erode_v,
                    2,
                    g_size,
                    l_size,
                    aux::DEBUG);
        }

        for (int i = 0; i < refine; i++) {
            dilate_h_events[i] = xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_dilate_h,
                    2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            dilate_v_events[i] = xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_dilate_v,
                    2,
                    g_size,
                    l_size,
                    aux::DEBUG);
        }

        maks_apply_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_power_apply,
                2,
                g_size,
                l_size,
                aux::DEBUG);


        // ======= KERNEL QUEUE FINALIZATION !
        xm::ocl::finish_queue(queue);


        // ======= KERNEL EXECUTION TIME MEASURING !
        if (aux::DEBUG) {
            if (maks_range_event != nullptr)
                Kernels::instance().print_time(xm::ocl::measure_exec_time(maks_range_event), "power_mask");
            for (int i = 0; i < refine; i++) {
                if (erode_h_events[i] != nullptr)
                    Kernels::instance().print_time(xm::ocl::measure_exec_time(erode_h_events[i]), "erode_h");
                if (erode_v_events[i] != nullptr)
                    Kernels::instance().print_time(xm::ocl::measure_exec_time(erode_v_events[i]), "erode_v");
            }
            for (int i = 0; i < refine; i++) {
                if (dilate_h_events[i] != nullptr)
                    Kernels::instance().print_time(xm::ocl::measure_exec_time(dilate_h_events[i]), "dilate_h");
                if (dilate_v_events[i] != nullptr)
                    Kernels::instance().print_time(xm::ocl::measure_exec_time(dilate_v_events[i]), "dilate_v");
            }
            if (maks_apply_event != nullptr)
                Kernels::instance().print_time(xm::ocl::measure_exec_time(maks_apply_event), "power_apply");
        }

        // ======= RELEASE HEAP DATA !
        clReleaseMemObject(buffer_io_1);
        clReleaseMemObject(buffer_io_2);

        xm::ocl::release_event(maks_range_event);
        xm::ocl::release_event(maks_apply_event);

        for (int i = 0; i < refine; i++) {
            xm::ocl::release_event(erode_h_events[i]);
            xm::ocl::release_event(erode_v_events[i]);
            xm::ocl::release_event(dilate_h_events[i]);
            xm::ocl::release_event(dilate_v_events[i]);
        }

        delete[] erode_h_events;
        delete[] erode_v_events;
        delete[] dilate_h_events;
        delete[] dilate_v_events;

        // ======= RESULT !
        out = std::move(result);
    }

    void chroma_key_single_pass(const cv::UMat &in, cv::UMat &out, const cv::Scalar &hls_low, const cv::Scalar &hls_up,
                                const cv::Scalar &color, bool linear, int mask_size, int blur, int queue_index) {
        auto img = xm::ocl::iop::from_cv_umat(in,
                                              Kernels::instance().ocl_context,
                                              Kernels::instance().device_id,
                                              xm::ocl::ACCESS::RO);
        chroma_key_single_pass(img, hls_low, hls_up, color, linear, mask_size, blur, queue_index)
                .waitFor()
                .toUMat(out);
    }

    xm::ocl::iop::QueuePromise chroma_key_single_pass(const cv::UMat &in,
                                                      const cv::Scalar &hls_low,
                                                      const cv::Scalar &hls_up,
                                                      const cv::Scalar &color,
                                                      bool linear, int mask_size, int blur,
                                                      int queue_index) {
        auto img = xm::ocl::iop::from_cv_umat(in,
                                              Kernels::instance().ocl_context,
                                              Kernels::instance().device_id,
                                              xm::ocl::ACCESS::RO);
        return chroma_key_single_pass(img, hls_low, hls_up, color, linear, mask_size, blur, queue_index);
    }

    xm::ocl::iop::QueuePromise chroma_key_single_pass(const Image2D &in, const cv::Scalar &hls_low, const cv::Scalar &hls_up,
                                                      const cv::Scalar &color, bool linear, int mask_size, int blur,
                                                      int queue_index) {
        const auto kernel_blur_buffer = Kernels::instance().blur_kernels[(blur - 1) / 2];
        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (int) ((float) n_w / ratio);

        const auto context = in.context;
        const auto queue = Kernels::instance().retrieve_queue(queue_index);
        const auto inter_size = in.cols * in.rows * 3;
        const auto pref_size = Kernels::instance().power_chroma_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(n_w, pref_size),
                            xm::ocl::optimal_global_size(n_h, pref_size)};

        // resize -> (blur_h -> blur_v) -> range_hls -> mask_apply
        cl_int err;

        cl_mem buffer_in = in.handle;
        cl_mem buffer_blur = Kernels::instance().blur_kernels[(blur - 1) / 2].handle;
        cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);

        auto kernel_chroma = Kernels::instance().kernel_power_chroma;

        auto blur_kern_half_size = (int) (blur / 2);
        auto mask_height = (uint) n_h;
        auto mask_width = (uint) n_w;
        auto out_height = (uint) in.rows;
        auto out_width = (uint) in.cols;
        auto scale_h = (float) in.rows / (float) n_h;
        auto scale_w = (float) in.cols / (float) n_w;
        auto lower_h = (uchar) hls_low[0];
        auto lower_l = (uchar) hls_low[1];
        auto lower_s = (uchar) hls_low[2];
        auto upper_h = (uchar) hls_up[0];
        auto upper_l = (uchar) hls_up[1];
        auto upper_s = (uchar) hls_up[2];
        auto color_b = (uchar) color[0];
        auto color_g = (uchar) color[1];
        auto color_r = (uchar) color[2];
        auto is_linear = (uchar) linear ? 1 : 0;
        auto is_blur = (uchar) (blur >= 3);
        auto dx = (uint) std::ceil(scale_w);
        auto dy = (uint) std::ceil(scale_h);

        xm::ocl::set_kernel_arg(kernel_chroma, 0, sizeof(cl_mem), &buffer_in);
        xm::ocl::set_kernel_arg(kernel_chroma, 1, sizeof(cl_mem), &buffer_out);

        xm::ocl::set_kernel_arg(kernel_chroma, 2, sizeof(cl_mem), &buffer_blur);
        xm::ocl::set_kernel_arg(kernel_chroma, 3, sizeof(int), &blur_kern_half_size);
        xm::ocl::set_kernel_arg(kernel_chroma, 4, sizeof(uchar), &is_blur);

        xm::ocl::set_kernel_arg(kernel_chroma, 5, sizeof(uchar), &is_linear);
        xm::ocl::set_kernel_arg(kernel_chroma, 6, sizeof(uint), &out_width);
        xm::ocl::set_kernel_arg(kernel_chroma, 7, sizeof(uint), &out_height);
        xm::ocl::set_kernel_arg(kernel_chroma, 8, sizeof(uint), &mask_width);
        xm::ocl::set_kernel_arg(kernel_chroma, 9, sizeof(uint), &mask_height);
        xm::ocl::set_kernel_arg(kernel_chroma, 10, sizeof(float), &scale_w);
        xm::ocl::set_kernel_arg(kernel_chroma, 11, sizeof(float), &scale_h);

        xm::ocl::set_kernel_arg(kernel_chroma, 12, sizeof(uchar), &lower_h);
        xm::ocl::set_kernel_arg(kernel_chroma, 13, sizeof(uchar), &lower_l);
        xm::ocl::set_kernel_arg(kernel_chroma, 14, sizeof(uchar), &lower_s);
        xm::ocl::set_kernel_arg(kernel_chroma, 15, sizeof(uchar), &upper_h);
        xm::ocl::set_kernel_arg(kernel_chroma, 16, sizeof(uchar), &upper_l);
        xm::ocl::set_kernel_arg(kernel_chroma, 17, sizeof(uchar), &upper_s);

        xm::ocl::set_kernel_arg(kernel_chroma, 18, sizeof(uchar), &color_b);
        xm::ocl::set_kernel_arg(kernel_chroma, 19, sizeof(uchar), &color_g);
        xm::ocl::set_kernel_arg(kernel_chroma, 20, sizeof(uchar), &color_r);

        xm::ocl::set_kernel_arg(kernel_chroma, 21, sizeof(uint), &dx);
        xm::ocl::set_kernel_arg(kernel_chroma, 22, sizeof(uint), &dy);

        cl_event chroma_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_chroma,
                2,
                g_size,
                l_size,
                aux::DEBUG);

        return xm::ocl::iop::QueuePromise(xm::ocl::Image2D(in, buffer_out), queue, chroma_event);
    }

}
#pragma clang diagnostic pop