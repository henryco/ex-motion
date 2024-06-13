//
// Created by henryco on 4/19/24.
//

#include <utility>
#include <opencv2/core/ocl.hpp>

#include "../../xmotion/core/camera/stereo_camera.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/ocl/ocl_interop.h"
#include "../../xmotion/core/ocl/cl_kernel.h"

namespace xm {
    int fourCC(const char *name) {
        return cv::VideoWriter::fourcc(
                name[0],
                name[1],
                name[2],
                name[3]);
    }

    StereoCamera::~StereoCamera() {
        for (auto &queue: command_queues) {
            if (queue.second == nullptr)
                continue;
            clReleaseCommandQueue(queue.second);
        }
        for (auto &capture: captures)
            capture.second.release();
        command_queues.clear();
        captures.clear();
    }

    void StereoCamera::release() {
        for (auto &capture: captures)
            capture.second.release();
        captures.clear();
    }

    void StereoCamera::open(const SCamProp &prop) {
        properties.push_back(prop);

        auto device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        auto ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
        command_queues[prop.device_id] = xm::ocl::create_queue_device(ocl_context, device_id, true, false);

        if (captures.contains(prop.device_id) && captures.at(prop.device_id).isOpened()) {
            log->warn("capture: {} is already open", prop.device_id);
            return;
        }

        if (prop.width <= 0 || prop.height <=0)
            throw std::runtime_error("Frame width or height cannot be <= 0 for device: " + prop.name);

        const auto api = platform::cap::video_capture_api();
        const auto idx = platform::cap::index_from_id(prop.device_id);

        std::vector<int> params;
        params.assign({
                              cv::CAP_PROP_FOURCC, fourCC(prop.codec.c_str()),
                              cv::CAP_PROP_FRAME_WIDTH, prop.width,
                              cv::CAP_PROP_FRAME_HEIGHT, prop.height,
                              cv::CAP_PROP_FPS, prop.fps,
                              cv::CAP_PROP_BUFFERSIZE, prop.buffer
                      });

        captures[prop.device_id] = cv::VideoCapture(idx, api, params);
    }

    std::map<std::string, xm::ocl::Image2D> StereoCamera::captureWithName() {
        if (captures.empty()) {
            log->warn("StereoCamera is not initialized");
            return {};
        }

        if (!executor) {
            log->debug("no active executors, creating one");

            // there is no assigned executors, create one
            executor = std::make_shared<eox::util::ThreadPool>();
            executor->start(captures.size());
        }

        std::vector<cv::VideoCapture> cameras;
        std::vector<int> ready;
        cameras.reserve(captures.size());
        ready.reserve(captures.size());
        for (auto &cam: captures) {
            cameras.push_back(cam.second);
        }

        if (fast) {
            // faster because it calls for buffer often, but less synchronized method of grabbing frames
            if (!cv::VideoCapture::waitAny(cameras, ready))
                return {};
        } else {
            // slower, but more precise (synchronized) method of grabbing frames
            for (auto &item: cameras) {
                if (!item.grab())
                    return {};
            }
        }

        std::vector<std::future<std::pair<std::string, xm::ocl::Image2D>>> results;
        results.reserve(captures.size());
        for (auto &capture: captures) {
            results.push_back(executor->execute<std::pair<std::string, xm::ocl::Image2D>>(
                    [&capture, this]() mutable -> std::pair<std::string, xm::ocl::Image2D> {
                        cv::Mat frame;
                        capture.second.retrieve(frame);
                        auto queue = command_queues.at(capture.first);
                        return {capture.first,
                                xm::ocl::iop::from_cv_mat(frame, queue).waitFor().getImage2D()};
                    }));
        }

        std::map<std::string, xm::ocl::Image2D> frames;
        for (auto &future: results) {
            auto pair = future.get();
            if (pair.second.empty()) {
                log->warn("empty frame");
                return {};
            }
            frames[pair.first] = pair.second;
        }

        // CROPPING AND FLIPPING
        std::map<std::string, xm::ocl::Image2D> images;
        std::map<std::string, xm::ocl::iop::ClImagePromise> promises;
        int queue_idx = 0;

        const auto t0 = std::chrono::system_clock::now();

        for (const auto &property: properties) {
            queue_idx += 1;

            const auto &src = frames.at(property.device_id);
            auto queue = command_queues.at(property.device_id);

            xm::ocl::Image2D dst;

            // whole frame
            if (property.x == 0 && property.y == 0 && property.w == property.width && property.h == property.height) {
//                xm::ocl::iop::copy_ocl(src, queue).waitFor().toImage2D(dst);
                dst = src;
            }
            // sub region
            else {
                xm::ocl::iop::copy_ocl(src, queue, property.x, property.y, property.w, property.h)
                .waitFor().toImage2D(dst);
            }

            if (property.flip_x || property.flip_y || property.rotate) {
                auto promise = xm::ocl::flip_rotate(dst, property.flip_x, property.flip_y, property.rotate, queue_idx);
                promises[property.name] = promise;
                continue; // we will iterate promises later
            }

            images[property.name] = dst;
        }

        for (auto &p: promises) {
            images[p.first] = p.second.waitFor().getImage2D();
        }

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        log->info("camera time: {}", d);

        return images;
    }

    std::vector<xm::ocl::Image2D> StereoCamera::capture() {
        const auto results = captureWithName();
        std::vector<xm::ocl::Image2D> vec;
        vec.reserve(properties.size());
        for (const auto &prop: properties)
            vec.push_back(results.at(prop.name));
        return vec;
    }

    std::vector<platform::cap::camera_controls> StereoCamera::getControls() const {
        std::vector<platform::cap::camera_controls> vec;
        vec.reserve(captures.size());
        for (const auto &item: captures)
            vec.emplace_back(platform::cap::query_controls(item.first));
        return vec;
    }

    platform::cap::camera_controls StereoCamera::getControls(const std::string &device_id) const {
        if (!captures.contains(device_id))
            throw std::runtime_error("No such device: " + device_id);
        return platform::cap::query_controls(device_id);
    }

    void StereoCamera::setControl(const std::string &device_id, uint prop_id, int value) {
        if (!captures.contains(device_id))
            throw std::runtime_error("No such device: " + device_id);
        platform::cap::set_control_value(device_id, prop_id, value);
    }

    void StereoCamera::resetControls(const std::string &device_id) {
        const auto controls = platform::cap::query_controls(device_id).controls;
        for (const auto &control: controls)
            setControl(device_id, control.id, control.default_value);
    }

    void StereoCamera::resetControls() {
        for (const auto &capture: captures)
            resetControls(capture.first);
    }

    void StereoCamera::save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const {
        try {
            const auto controls = getControls(device_id);
            platform::cap::save(output_stream, name, controls);
        } catch (std::exception &e) {
            log->warn("Error during capture device save: {}", e.what());
            return;
        } catch (...) {
            log->warn("Unexpected error during capture device save");
        }
    }

    void StereoCamera::read(std::istream &input_stream, const std::string &device_id, const std::string &name) {
        try {
            const auto controls = platform::cap::read(input_stream, name);
            resetControls(device_id);
            for (const auto &control: controls.controls)
                setControl(device_id, control.id, control.value);
        } catch (std::exception &e) {
            log->warn("Error during capture device read: {}", e.what());
            return;
        } catch (...) {
            log->warn("Unexpected error during capture device read");
        }
    }

    void StereoCamera::setFastMode(bool _fast) {
        fast = _fast;
    }

    bool StereoCamera::getFastMode() const {
        return fast;
    }

    void StereoCamera::setThreadPool(std::shared_ptr<eox::util::ThreadPool> _executor) {
        this->executor = std::move(_executor);
    }

    uint StereoCamera::getDeviceIndex(const std::string &device_id) const {
        return platform::cap::index_from_id(device_id);
    }
}