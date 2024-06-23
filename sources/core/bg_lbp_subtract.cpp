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
//            model_i += 1;
            return frame_in;
        }

        return frame_in;
    }

    void BgLbpSubtract::prepare_update_model(const ocl::iop::ClImagePromise &in_p, int q_idx) {
        const auto &in = in_p.getImage2D();
        size_t l_size[2] = {pref_size, pref_size};
        size_t g_size[2] = {xm::ocl::optimal_global_size((int) in.cols, pref_size),
                            xm::ocl::optimal_global_size((int) in.rows, pref_size)};



        cl_int err;
    }

}