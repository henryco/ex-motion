//
// Created by henryco on 5/10/24.
//

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/camera/d_dummy_camera.h"
#include "../../xmotion/core/ocl/ocl_interop.h"

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

std::map<std::string, xm::ocl::Image2D> xm::DummyCamera::captureWithName() {
    std::map<std::string, xm::ocl::Image2D> map;
    for (auto & prop : properties) {
        cv::UMat cpy;
        src.copyTo(cpy);
        map[prop.name] = xm::ocl::iop::from_cv_umat(cpy);
    }
    return map;
}

std::vector<xm::ocl::Image2D> xm::DummyCamera::capture() {
    std::vector<xm::ocl::Image2D> images;
    images.reserve(properties.size());
    for (int i = 0; i < properties.size(); i++) {
        cv::UMat cpy;
        src.copyTo(cpy);
        images.push_back(xm::ocl::iop::from_cv_umat(cpy));
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
