//
// Created by henryco on 6/3/24.
//

#include <iostream>
#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"
#pragma ide diagnostic ignored "UnusedLocalVariable"
#pragma ide diagnostic ignored "bugprone-easily-swappable-parameters"

#include "../../xmotion/core/ocl/ocl_filters.h"

#include "../../kernels/chroma_key.h"
#include "../../kernels/flip_rotate.h"
#include "../../kernels/color_space.h"
#include "../../kernels/filter_conv.h"
#include "../../kernels/background.h"

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

        ocl_command_queue = xm::ocl::create_queue_device(ocl_context, device_id, true, aux::DEBUG);

        program_filter_conv = xm::ocl::build_program(ocl_context, device_id,
                                                     ocl_kernel_filter_conv_data,
                                                     ocl_kernel_filter_conv_data_size);
        program_color_space = xm::ocl::build_program(ocl_context, device_id,
                                                     ocl_kernel_color_space_data,
                                                     ocl_kernel_color_space_data_size);
        program_power_chroma = xm::ocl::build_program(ocl_context, device_id,
                                                      ocl_kernel_chroma_key_data,
                                                      ocl_kernel_chroma_key_data_size);
        program_flip_rotate = xm::ocl::build_program(ocl_context, device_id,
                                                     ocl_kernel_flip_rotate_data,
                                                     ocl_kernel_flip_rotate_data_size);
        program_background = xm::ocl::build_program(ocl_context, device_id,
                                                    ocl_kernel_background_data,
                                                    ocl_kernel_background_data_size);

        kernel_blur_h = xm::ocl::build_kernel(program_filter_conv, "gaussian_blur_horizontal");
        kernel_blur_v = xm::ocl::build_kernel(program_filter_conv, "gaussian_blur_vertical");
        blur_local_size = xm::ocl::optimal_local_size(device_id, kernel_blur_h);

        kernel_dilate_h = xm::ocl::build_kernel(program_filter_conv, "dilate_horizontal");
        kernel_dilate_v = xm::ocl::build_kernel(program_filter_conv, "dilate_vertical");
        dilate_local_size = xm::ocl::optimal_local_size(device_id, kernel_dilate_h);

        kernel_erode_h = xm::ocl::build_kernel(program_filter_conv, "erode_horizontal");
        kernel_erode_v = xm::ocl::build_kernel(program_filter_conv, "erode_vertical");
        erode_local_size = xm::ocl::optimal_local_size(device_id, kernel_erode_h);

        kernel_range_hls = xm::ocl::build_kernel(program_color_space, "kernel_range_hls_mask");
        range_hls_local_size = xm::ocl::optimal_local_size(device_id, kernel_range_hls);

        kernel_mask_apply = xm::ocl::build_kernel(program_color_space, "kernel_simple_mask_apply");
        mask_apply_local_size = xm::ocl::optimal_local_size(device_id, kernel_mask_apply);

        kernel_power_chroma = xm::ocl::build_kernel(program_power_chroma, "power_chromakey");
        kernel_power_apply = xm::ocl::build_kernel(program_power_chroma, "power_apply");
        kernel_power_mask = xm::ocl::build_kernel(program_power_chroma, "power_mask");
        power_chroma_local_size = xm::ocl::optimal_local_size(device_id, kernel_power_chroma);

        kernel_flip_rotate = xm::ocl::build_kernel(program_flip_rotate, "flip_rotate");
        flip_rotate_local_size = xm::ocl::optimal_local_size(device_id, kernel_flip_rotate);

        kernel_lbp_texture = xm::ocl::build_kernel(program_background, "kernel_lbp");
        kernel_lbp_mask_only = xm::ocl::build_kernel(program_background, "kernel_mask_only");
        kernel_lbp_mask_apply = xm::ocl::build_kernel(program_background, "kernel_mask_apply");
        kernel_lbp_power = xm::ocl::build_kernel(program_background, "kernel_lbp_mask_apply");
        lbp_local_size = xm::ocl::optimal_local_size(device_id, kernel_lbp_power);

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
        clReleaseKernel(kernel_dilate_h);
        clReleaseKernel(kernel_dilate_v);
        clReleaseKernel(kernel_erode_h);
        clReleaseKernel(kernel_erode_v);
        clReleaseProgram(program_filter_conv);

        clReleaseKernel(kernel_range_hls);
        clReleaseKernel(kernel_mask_apply);
        clReleaseProgram(program_color_space);

        clReleaseKernel(kernel_power_chroma);
        clReleaseKernel(kernel_power_apply);
        clReleaseKernel(kernel_power_mask);
        clReleaseProgram(program_power_chroma);

        clReleaseKernel(kernel_flip_rotate);
        clReleaseProgram(program_flip_rotate);

        clReleaseKernel(kernel_lbp_texture);
        clReleaseKernel(kernel_lbp_mask_only);
        clReleaseKernel(kernel_lbp_mask_apply);
        clReleaseKernel(kernel_lbp_power);
        clReleaseProgram(program_background);

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


    xm::ocl::iop::ClImagePromise blur(const Image2D &in, int kernel_size, int queue_index) {
        return blur(Kernels::instance().retrieve_queue(queue_index), in, kernel_size);
    }

    xm::ocl::iop::ClImagePromise blur(cl_command_queue queue, const Image2D &in, int kernel_size) {
        if (kernel_size < 3 || kernel_size % 2 == 0 || kernel_size > 31)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        const auto context = Kernels::instance().ocl_context;
        const auto pref_size = Kernels::instance().blur_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) in.cols, pref_size),
                            xm::ocl::optimal_global_size((int) in.rows, pref_size)};

        cl_int err;

        cl_mem buffer_in = in.handle;
        cl_mem kernel_mat_buffer = Kernels::instance().blur_kernels[(kernel_size - 1) / 2].handle;
        cl_mem buffer_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, in.size(), NULL, &err);
        cl_mem buffer_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, in.size(), NULL, &err);

        auto width = (uint) in.cols;
        auto height = (uint) in.rows;
        auto kh_size = (int) (kernel_size / 2);

        auto kernel_h = Kernels::instance().kernel_blur_h;
        auto kernel_v = Kernels::instance().kernel_blur_v;

        xm::ocl::set_kernel_arg(kernel_h, 0, sizeof(cl_mem), &buffer_in);
        xm::ocl::set_kernel_arg(kernel_h, 1, sizeof(cl_mem), &kernel_mat_buffer);
        xm::ocl::set_kernel_arg(kernel_h, 2, sizeof(cl_mem), &buffer_1);
        xm::ocl::set_kernel_arg(kernel_h, 3, sizeof(uint), &width);
        xm::ocl::set_kernel_arg(kernel_h, 4, sizeof(uint), &height);
        xm::ocl::set_kernel_arg(kernel_h, 5, sizeof(int), &kh_size);

        xm::ocl::set_kernel_arg(kernel_v, 0, sizeof(cl_mem), &buffer_1);
        xm::ocl::set_kernel_arg(kernel_v, 1, sizeof(cl_mem), &kernel_mat_buffer);
        xm::ocl::set_kernel_arg(kernel_v, 2, sizeof(cl_mem), &buffer_2);
        xm::ocl::set_kernel_arg(kernel_v, 3, sizeof(uint), &width);
        xm::ocl::set_kernel_arg(kernel_v, 4, sizeof(uint), &height);
        xm::ocl::set_kernel_arg(kernel_v, 5, sizeof(int), &kh_size);

        xm::ocl::enqueue_kernel_fast(
                queue, kernel_h, 2,
                g_size,
                l_size);

        xm::ocl::enqueue_kernel_fast(
                queue, kernel_v, 2,
                g_size,
                l_size);

        xm::ocl::Image2D image(in, buffer_2);
        return xm::ocl::iop::ClImagePromise(image, queue)
        .withCleanup(new std::function<void()>([buffer_1]() {
            clReleaseMemObject(buffer_1);
        }));
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

    xm::ocl::iop::ClImagePromise chroma_key(cl_command_queue queue, const Image2D &in, const xm::ds::Color4u &hls_low, const xm::ds::Color4u &hls_up,
                                            const xm::ds::Color4u &color, bool linear, int mask_size, int blur, int fine, int refine) {
        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (int) ((float) n_w / ratio);

        // power_mask -> (erode_h -> erode_v) -> (dilate_h -> dilate_v) -> mask_apply

        cl_int err;

        const auto context = Kernels::instance().ocl_context;
        const auto inter_size = n_w * n_h * 1; // grayscale/black-white mask, only one channel
        const auto pref_size = Kernels::instance().mask_apply_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(n_w, pref_size),
                            xm::ocl::optimal_global_size(n_h, pref_size)};

        // ======= BUFFERS ALLOCATION !
        cl_mem buffer_in = (cl_mem) in.handle;
        cl_mem buffer_blur = (cl_mem) Kernels::instance().blur_kernels[(blur - 1) / 2].handle;
        cl_mem buffer_io_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);
        cl_mem buffer_io_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);
        cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, in.size(), NULL, &err);


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
        auto lower_h = (uchar) hls_low.h;
        auto lower_l = (uchar) hls_low.l;
        auto lower_s = (uchar) hls_low.s;
        auto upper_h = (uchar) hls_up.h;
        auto upper_l = (uchar) hls_up.l;
        auto upper_s = (uchar) hls_up.s;
        auto color_b = (uchar) color.b;
        auto color_g = (uchar) color.g;
        auto color_r = (uchar) color.r;
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

        // ======= KERNEL ENQUEUE !
        xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_power_mask,
                2,
                g_size,
                l_size,
                false);

        for (int i = 0; i < refine; i++) {
            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_erode_h,
                    2,
                    g_size,
                    l_size,
                    false);
            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_erode_v,
                    2,
                    g_size,
                    l_size,
                    false);
        }

        for (int i = 0; i < refine; i++) {
            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_dilate_h,
                    2,
                    g_size,
                    l_size,
                    false);
            xm::ocl::enqueue_kernel_fast(
                    queue,
                    kernel_dilate_v,
                    2,
                    g_size,
                    l_size,
                    false);
        }

        xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_power_apply,
                2,
                g_size,
                l_size,
                false);

        return xm::ocl::iop::ClImagePromise(xm::ocl::Image2D(in, buffer_out), queue)
        .withCleanup(new std::function<void()>([buffer_io_1, buffer_io_2]() {
            clReleaseMemObject(buffer_io_1);
            clReleaseMemObject(buffer_io_2);
        }));
    }

    xm::ocl::iop::ClImagePromise chroma_key(const Image2D &in, const xm::ds::Color4u &hls_low, const xm::ds::Color4u &hls_up, const xm::ds::Color4u &color,
                    bool linear, int mask_size, int blur, int fine, int refine, int queue_index) {
        return chroma_key(Kernels::instance().retrieve_queue(queue_index), in, hls_low, hls_up, color, linear, mask_size, blur, fine, refine);
    }

    xm::ocl::iop::ClImagePromise chroma_key_single_pass(cl_command_queue queue, const Image2D &in, const xm::ds::Color4u &hls_low,
                                                        const xm::ds::Color4u &hls_up, const xm::ds::Color4u &color, bool linear, int mask_size,
                                                        int blur) {
        const auto kernel_blur_buffer = Kernels::instance().blur_kernels[(blur - 1) / 2];
        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (int) ((float) n_w / ratio);

        const auto context = in.context;
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
        auto lower_h = (uchar) hls_low.h;
        auto lower_l = (uchar) hls_low.l;
        auto lower_s = (uchar) hls_low.s;
        auto upper_h = (uchar) hls_up.h;
        auto upper_l = (uchar) hls_up.l;
        auto upper_s = (uchar) hls_up.s;
        auto color_b = (uchar) color.b;
        auto color_g = (uchar) color.g;
        auto color_r = (uchar) color.r;
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

        return xm::ocl::iop::ClImagePromise(xm::ocl::Image2D(in, buffer_out), queue, chroma_event);
    }

    xm::ocl::iop::ClImagePromise chroma_key_single_pass(const Image2D &in, const xm::ds::Color4u &hls_low, const xm::ds::Color4u &hls_up,
                                                        const xm::ds::Color4u &color, bool linear, int mask_size, int blur,
                                                        int queue_index) {
        return chroma_key_single_pass(Kernels::instance().retrieve_queue(queue_index),
                                      in, hls_low, hls_up, color, linear, mask_size, blur);
    }

    xm::ocl::iop::ClImagePromise flip_rotate(const Image2D &in, bool flip_x, bool flip_y, bool rotate, int queue_index) {
        return flip_rotate(Kernels::instance().retrieve_queue(queue_index), in, flip_x, flip_y, rotate);
    }

    xm::ocl::iop::ClImagePromise flip_rotate(cl_command_queue queue, const Image2D &in, bool flip_x, bool flip_y, bool rotate) {
        const auto context = in.context;
        const auto kernel = Kernels::instance().kernel_flip_rotate;
        const auto pref_size = Kernels::instance().flip_rotate_local_size;
        const auto inter_size = in.cols * in.rows * in.channels;

        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) in.cols, pref_size),
                            xm::ocl::optimal_global_size((int) in.rows, pref_size)};

        cl_int err;
        cl_mem buffer_in = in.handle;
        cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Cannot create cl buffer: " + std::to_string(err));

        auto width = (int) in.cols;
        auto height = (int) in.rows;
        auto c_size = (int) in.channels;
        auto _x = (int) (flip_x ? 1 : 0);
        auto _y = (int) (flip_y ? 1 : 0);
        auto _r = (int) (rotate ? 1 : 0);

        xm::ocl::set_kernel_arg(kernel, 0, sizeof(cl_mem), &buffer_in);
        xm::ocl::set_kernel_arg(kernel, 1, sizeof(cl_mem), &buffer_out);
        xm::ocl::set_kernel_arg(kernel, 2, sizeof(int), &width);
        xm::ocl::set_kernel_arg(kernel, 3, sizeof(int), &height);
        xm::ocl::set_kernel_arg(kernel, 4, sizeof(int), &c_size);
        xm::ocl::set_kernel_arg(kernel, 5, sizeof(int), &_x);
        xm::ocl::set_kernel_arg(kernel, 6, sizeof(int), &_y);
        xm::ocl::set_kernel_arg(kernel, 7, sizeof(int), &_r);

        cl_event flip_rotate_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel,
                2,
                g_size,
                l_size,
                aux::DEBUG);

        return xm::ocl::iop::ClImagePromise(xm::ocl::Image2D(
                                                    rotate ? height : width,
                                                    rotate ? width : height,
                                                    in.channels, in.channel_size,
                                                    buffer_out, in.context, in.device, xm::ocl::ACCESS::RW),
                                            queue, flip_rotate_event);
    }

    xm::ocl::iop::ClImagePromise local_binary_patterns(const Image2D &in, int window_size, int queue_index) {
        return local_binary_patterns(Kernels::instance().retrieve_queue(queue_index), in, window_size);
    }

    xm::ocl::iop::ClImagePromise local_binary_patterns(cl_command_queue queue, const Image2D &in, int window_size) {
        if (window_size > 15)
            throw std::invalid_argument("window_size > 15");

        auto c_size = (int) std::ceil((float) (std::pow(window_size, 2) - 1) / 8.f);

        if (c_size < 1)
            throw std::invalid_argument("output channels size < 1");

        const auto context = in.context;
        const auto inter_size = in.cols * in.rows * c_size;
        const auto pref_size = Kernels::instance().lbp_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) in.cols, pref_size),
                            xm::ocl::optimal_global_size((int) in.rows, pref_size)};

        cl_int err;

        cl_mem buffer_in = in.handle;
        cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);

        auto kernel_lbp = Kernels::instance().kernel_lbp_texture;

        auto i_size = (int) in.channels;
        auto k_size = (int) window_size;
        auto width = (int) in.cols;
        auto height = (int) in.rows;

        xm::ocl::set_kernel_arg(kernel_lbp, 0, sizeof(cl_mem), &buffer_in);
        xm::ocl::set_kernel_arg(kernel_lbp, 1, sizeof(cl_mem), &buffer_out);
        xm::ocl::set_kernel_arg(kernel_lbp, 2, sizeof(int), &i_size);
        xm::ocl::set_kernel_arg(kernel_lbp, 3, sizeof(int), &c_size);
        xm::ocl::set_kernel_arg(kernel_lbp, 4, sizeof(int), &k_size);
        xm::ocl::set_kernel_arg(kernel_lbp, 5, sizeof(int), &width);
        xm::ocl::set_kernel_arg(kernel_lbp, 6, sizeof(int), &height);

        cl_event lbp_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_lbp,
                2,
                g_size,
                l_size,
                aux::DEBUG);

        xm::ocl::Image2D image(in.cols, in.rows, c_size, 1, buffer_out, in.context, in.device, xm::ocl::ACCESS::RW);
        return xm::ocl::iop::ClImagePromise(image, queue, lbp_event);
    }

    xm::ocl::iop::ClImagePromise subtract_bg_lbp_single_pass(
            const Image2D &lbp_texture, const Image2D &frame, const ds::Color4u &color,
            float threshold, int window_size, int queue_index) {
        return subtract_bg_lbp_single_pass(Kernels::instance().retrieve_queue(queue_index), lbp_texture, frame, color,
                                           threshold, window_size);
    }

    xm::ocl::iop::ClImagePromise subtract_bg_lbp_single_pass(
            cl_command_queue queue, const Image2D &lbp_texture, const Image2D &frame,
            const ds::Color4u &color, float threshold, int window_size) {
        if (window_size > 15)
            throw std::invalid_argument("window_size > 15");

        auto c_size = (int) std::ceil((float) (std::pow(window_size, 2) - 1) / 8.f);

        if (c_size < 1)
            throw std::invalid_argument("output channels size < 1");

        const auto context = frame.context;
        const auto inter_size = frame.size();
        const auto pref_size = Kernels::instance().lbp_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) frame.cols, pref_size),
                            xm::ocl::optimal_global_size((int) frame.rows, pref_size)};

        cl_int err;

        cl_mem buffer_in = frame.handle;
        cl_mem buffer_tex = lbp_texture.handle;
        cl_mem buffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, inter_size, NULL, &err);

        auto kernel_lbp_power = Kernels::instance().kernel_lbp_power;

        auto i_size = (int) frame.channels;
        auto k_size = (int) window_size;
        auto width = (int) frame.cols;
        auto height = (int) frame.rows;
        auto total = (int) std::pow(window_size, 2) - 1;
        auto color_b = (uint) color.b;
        auto color_g = (uint) color.g;
        auto color_r = (uint) color.r;

        xm::ocl::set_kernel_arg(kernel_lbp_power, 0, sizeof(cl_mem), &buffer_in);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 1, sizeof(cl_mem), &buffer_tex);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 2, sizeof(cl_mem), &buffer_out);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 3, sizeof(int), &i_size);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 4, sizeof(int), &c_size);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 5, sizeof(int), &k_size);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 6, sizeof(int), &total);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 7, sizeof(float), &threshold);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 8, sizeof(int), &width);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 9, sizeof(int), &height);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 10, sizeof(uchar), &color_b);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 11, sizeof(uchar), &color_g);
        xm::ocl::set_kernel_arg(kernel_lbp_power, 12, sizeof(uchar), &color_r);

        cl_event lbp_power_event = xm::ocl::enqueue_kernel_fast(
                queue,
                kernel_lbp_power,
                2,
                g_size,
                l_size,
                aux::DEBUG);

        return xm::ocl::iop::ClImagePromise(xm::ocl::Image2D(frame, buffer_out),
                                            queue,
                                            lbp_power_event);
    }
}
#pragma clang diagnostic pop