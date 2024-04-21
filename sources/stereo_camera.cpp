//
// Created by henryco on 4/19/24.
//

#include <utility>

#include "../xmotion/camera/stereo_camera.h"

namespace xm {
    int fourCC(const char *name) {
        return cv::VideoWriter::fourcc(
                name[0],
                name[1],
                name[2],
                name[3]);
    }

    StereoCamera::~StereoCamera() {
        release();
    }

    void StereoCamera::release() {
        for (auto &capture: captures)
            capture.second.release();
        captures.clear();
    }

    void StereoCamera::open(const SCamProp &prop) {

        auto cpy = prop; // TODO: RECALL HOW COPY/REFERENCES work in c++
        name_props[prop.name] = cpy;

        if (captures.contains(prop.device_id) && captures.at(prop.device_id).isOpened()) {
            log->warn("capture: {} is already open", prop.device_id);
            return;
        }

        const auto api = xm::cap::video_capture_api();
        const auto idx = xm::cap::index_from_id(prop.device_id);

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

    std::map<std::string, cv::Mat> StereoCamera::captureWithName() {
        std::map<std::string, cv::Mat> frames;

        if (captures.empty()) {
            log->warn("StereoCamera is not initialized");
            return {};
        }

        if (!executor) {
            log->debug("no active executors, creating one");

            // there is no assigned executors
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

        std::vector<std::future<std::pair<std::string, cv::Mat>>> results;
        results.reserve(captures.size());
        for (auto &capture: captures) {
            results.push_back(executor->execute<std::pair<std::string, cv::Mat>>(
                    [&capture]() mutable -> std::pair<std::string, cv::Mat> {
                        cv::Mat frame;
                        capture.second.retrieve(frame);
                        return {capture.first, frame};
                    }));
        }

        // TODO: FLIPPING and CROPPING

        for (auto &future: results) {
            auto pair = future.get();
            if (pair.second.empty()) {
                log->warn("empty frame");
                return frames;
            }
            frames[pair.first] = pair.second;
        }

        return frames;
    }

    std::vector<cv::Mat> StereoCamera::capture() {
        const auto results = captureWithName();
        std::vector<cv::Mat> vec;
        vec.reserve(results.size());
        for (const auto &pair: results)
            vec.push_back(pair.second);
        return vec;
    }

    std::vector<xm::cap::camera_controls> StereoCamera::getControls() const {
        std::vector<xm::cap::camera_controls> vec;
        vec.reserve(captures.size());
        for (const auto &item: captures)
            vec.emplace_back(xm::cap::query_controls(item.first));
        return vec;
    }

    xm::cap::camera_controls StereoCamera::getControls(const std::string &device_id) const {
        if (!captures.contains(device_id))
            throw std::runtime_error("No such device: " + device_id);
        return xm::cap::query_controls(device_id);
    }

    void StereoCamera::setControl(const std::string &device_id, uint prop_id, int value) {
        if (!captures.contains(device_id))
            throw std::runtime_error("No such device: " + device_id);
        xm::cap::set_control_value(device_id, prop_id, value);
    }

    void StereoCamera::resetControls(const std::string &device_id) {
        const auto controls = xm::cap::query_controls(device_id).controls;
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
            xm::cap::save(output_stream, name, controls);
        } catch (std::exception &e) {
            log->warn("Error during capture device save: {}", e.what());
            return;
        } catch (...) {
            log->warn("Unexpected error during capture device save");
        }
    }

    void StereoCamera::read(std::istream &input_stream, const std::string &device_id, const std::string &name) {
        try {
            const auto controls = xm::cap::read(input_stream, name);
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
        return xm::cap::index_from_id(device_id);
    }
}