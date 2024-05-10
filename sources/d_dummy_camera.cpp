//
// Created by henryco on 5/10/24.
//

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include "../xmotion/camera/d_dummy_camera.h"

void xm::DummyCamera::release() {
    StereoCamera::release();
}

void xm::DummyCamera::open(const xm::SCamProp &prop) {
    auto cpy = prop; // TODO: RECALL HOW COPY/REFERENCES work in c++
    properties.push_back(cpy);

    auto image = cv::imread(std::filesystem::path("../media/pose2.png").string());
    cv::resize(image, image, cv::Size(prop.width, prop.height));
    images.push_back(image);
}

std::map<std::string, cv::Mat> xm::DummyCamera::captureWithName() {
    std::map<std::string, cv::Mat> map;
    for (int i = 0; i < properties.size(); i++)
        map[properties.at(i).name] = images.at(i);
    return map;
}

std::vector<cv::Mat> xm::DummyCamera::capture() {
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
