//
// Created by henryco on 5/10/24.
//

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/camera/d_dummy_camera.h"

void xm::DummyCamera::release() {
    StereoCamera::release();
}

void xm::DummyCamera::open(const xm::SCamProp &prop) {
    auto cpy = prop; // TODO: RECALL HOW COPY/REFERENCES work in c++
    properties.push_back(cpy);

    cv::UMat image;
    cv::imread(std::filesystem::path("../media/pose2.png").string()).copyTo(image);
    cv::resize(image, image, cv::Size(prop.width, prop.height));
    src = image;
}

std::map<std::string, cv::UMat> xm::DummyCamera::captureWithName() {
    std::map<std::string, cv::UMat> map;
    for (auto & prop : properties) {
        cv::UMat cpy;
        src.copyTo(cpy);
        map[prop.name] = cpy;
    }
    return map;
}

std::vector<cv::UMat> xm::DummyCamera::capture() {
    std::vector<cv::UMat> images;
    images.reserve(properties.size());
    for (int i = 0; i < properties.size(); i++) {
        cv::UMat cpy;
        src.copyTo(cpy);
        images.push_back(cpy);
    }
    return images;
}

void xm::DummyCamera::setControl(const std::string &device_id, uint prop_id, int value) {}

void xm::DummyCamera::resetControls(const std::string &device_id) {}

void xm::DummyCamera::resetControls() {}

void xm::DummyCamera::setFastMode(bool fast) {}

void xm::DummyCamera::setThreadPool(std::shared_ptr<eox::util::ThreadPool> executor) {}

void xm::DummyCamera::save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const {}

void xm::DummyCamera::read(std::istream &input_stream, const std::string &device_id, const std::string &name) {}

platform::cap::camera_controls xm::DummyCamera::getControls(const std::string &device_id) const {
    return {
        .id = device_id,
        .controls = {}
    };
}

std::vector<platform::cap::camera_controls> xm::DummyCamera::getControls() const {
    return {};
}

bool xm::DummyCamera::getFastMode() const {
    return false;
}
