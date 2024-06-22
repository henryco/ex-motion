//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/filter/bg_lbp_subtract.h"
#include "../../xmotion/core/ocl/ocl_filters.h"

namespace xm::filters {

    BgLbpSubtract::BgLbpSubtract() {
        device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        ocl_command_queue = xm::ocl::create_queue_device(
                ocl_context,
                device_id,
                true,
                false);
    }

    BgLbpSubtract::~BgLbpSubtract() {
        for (auto &item: ocl_queue_map) {
            if (item.second == nullptr)
                continue;
            clReleaseCommandQueue(item.second);
        }

        clReleaseCommandQueue(ocl_command_queue);
        clReleaseContext(ocl_context);
        clReleaseDevice(device_id);
    }

    void BgLbpSubtract::reset() {
        ready = false;
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
        threshold = std::clamp(conf.threshold, 0.f, 1.f);
        lbp_kernel = std::clamp(conf.window * 2 + 1, 3, 15);
        blur_kernel = std::max(0, conf.blur * 2 + 1);
        fine_kernel = std::max(3, conf.fine * 2 + 1);
        fine_iterations = std::max(0, conf.refine);
        delay = std::max(0L, conf.delay);
        bgr_bg_color = conf.color;

        initialized = true;
        reset();
    }

    xm::ocl::iop::ClImagePromise BgLbpSubtract::filter(const ocl::Image2D &frame_in, int q_idx) {
        if (!initialized)
            throw std::logic_error("Filter is not initialized");

        if (!ready) {
            // TODO
            return frame_in;
        }

        return frame_in;
    }

}