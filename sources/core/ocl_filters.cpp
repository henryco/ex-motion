//
// Created by henryco on 6/3/24.
//

#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/ocl/ocl_kernels.h"
#include "../../xmotion/core/ocl/cl_kernel.h"
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

        aux::DEBUG = true;

        std::lock_guard<std::mutex> lock(aux::GLOBAL_MUTEX);

        log->debug("[{}] oCL kernels initialization, profiling: {}", thread_id, aux::DEBUG);

//        device_id = xm::ocl::find_gpu_device();
//        ocl_context = xm::ocl::create_context(device_id);
//        svm_supported = check_svm_cap(device_id);

        device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        svm_supported = check_svm_cap(device_id);

        ocl_command_queue = xm::ocl::create_queue(ocl_context, device_id, aux::DEBUG);

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

        clReleaseCommandQueue(ocl_command_queue);
        clReleaseContext(ocl_context);
        clReleaseDevice(device_id);

        log->debug("[{}] released oCL kernels", thread_id);
    }

    void Kernels::print_time(cl_ulong time, const std::string &name) const {
        if (aux::DEBUG)
            log->debug("[{}] kernel [{}] execution time: {} ns", thread_id, name, std::to_string(time));
    }

    void blur(const cv::UMat &in, cv::UMat &out, const int kernel_size, float sigma) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        if (sigma <= 0)
            sigma = ((float) kernel_size - 1.f) / 6.f;

        cv::UMat result_1(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        cv::UMat kernel_mat;
        cv::getGaussianKernel(kernel_size, sigma, CV_32F).copyTo(kernel_mat);

        auto kernel_mat_buffer = (cl_mem) kernel_mat.handle(cv::ACCESS_READ);
        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().ocl_command_queue;
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

        out = result_2;
    }

    void bgr_in_range_hls(const cv::Scalar &hls_low, const cv::Scalar &hls_up, const cv::UMat &in, cv::UMat &out) {
        cv::UMat result(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat bot(1, 1, CV_8UC3, hls_low, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat top(1, 1, CV_8UC3, hls_up, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const auto queue = Kernels::instance().ocl_command_queue;
        const auto pref_size = Kernels::instance().range_hls_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(in.cols, pref_size),
                            xm::ocl::optimal_global_size(in.rows, pref_size)};

        {
            auto kernel = Kernels::instance().kernel_range_hls;
            auto buffer_in = (cl_mem) in.handle(cv::ACCESS_READ);
            auto buffer_top = (cl_mem) bot.handle(cv::ACCESS_READ);
            auto buffer_bot = (cl_mem) top.handle(cv::ACCESS_READ);
            auto buffer_out = (cl_mem) result.handle(cv::ACCESS_WRITE);
            auto width = (uint) in.cols;
            auto height = (uint) in.rows;

            xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_in);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_bot);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(cl_mem), &buffer_top);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(cl_mem), &buffer_out);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &width);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 5, sizeof(uint), &height);

            const auto time = xm::ocl::enqueue_kernel_sync(
                    queue, kernel, 2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            Kernels::instance().print_time(time, "range_hls");
        }

        out = result;
    }

    void dilate(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().ocl_command_queue;
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

        out = result_2;
    }

    void erode(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kh_size = (int) (kernel_size / 2);

        const auto queue = Kernels::instance().ocl_command_queue;
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

        out = result_2;
    }

    void apply_mask_with_color(const cv::Scalar &color, const cv::UMat &img, const cv::UMat &mask, cv::UMat &out) {
        cv::UMat result(img.rows, img.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat clr(1, 1, CV_8UC3, color, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const auto queue = Kernels::instance().ocl_command_queue;
        const auto pref_size = Kernels::instance().mask_apply_local_size;
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size(img.cols, pref_size),
                            xm::ocl::optimal_global_size(img.rows, pref_size)};

        {
            auto kernel = Kernels::instance().kernel_mask_apply;

            auto buffer_mask = (cl_mem) mask.handle(cv::ACCESS_READ);
            auto buffer_image = (cl_mem) img.handle(cv::ACCESS_READ);
            auto buffer_color = (cl_mem) clr.handle(cv::ACCESS_READ);
            auto buffer_result = (cl_mem) result.handle(cv::ACCESS_WRITE);
            auto mask_width = (uint) mask.cols;
            auto mask_height = (uint) mask.rows;
            auto width = (uint) img.cols;
            auto height = (uint) img.rows;
            auto s_w = (float) mask.cols / (float) img.cols;
            auto s_h = (float) mask.rows / (float) img.rows;

            xm::ocl::set_kernel_arg(kernel, (cl_uint) 0, sizeof(cl_mem), &buffer_mask);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 1, sizeof(cl_mem), &buffer_image);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 2, sizeof(cl_mem), &buffer_color);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 3, sizeof(cl_mem), &buffer_result);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 4, sizeof(uint), &mask_width);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 5, sizeof(uint), &mask_height);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 6, sizeof(uint), &width);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 7, sizeof(uint), &height);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 8, sizeof(cl_float), &s_w);
            xm::ocl::set_kernel_arg(kernel, (cl_uint) 9, sizeof(cl_float), &s_h);

            const auto time = xm::ocl::enqueue_kernel_sync(
                    queue, kernel, 2,
                    g_size,
                    l_size,
                    aux::DEBUG);
            Kernels::instance().print_time(time, "apply_mask");
        }

        out = result;
    }

}