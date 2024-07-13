//
// Created by henryco on 5/10/24.
//

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include "../../xmotion/core/camera/d_dummy_camera.h"
#include "../../xmotion/core/ocl/ocl_interop.h"
#include "../../xmotion/core/ocl/cl_kernel.h"

void xm::DummyCamera::release() {
    StereoCamera::release();
}

void xm::DummyCamera::open(const xm::SCamProp &prop) {
    auto cpy = prop; // TODO: RECALL HOW COPY/REFERENCES work in c++
    properties.push_back(cpy);

    cv::Mat image;
    cv::imread(std::filesystem::path("../media/pose2.png").string()).copyTo(image);
    cv::resize(image, image, cv::Size(prop.width, prop.height));
    src = image;

    auto device_id = (cl_device_id) cv::ocl::Device::getDefault().ptr();
    auto ocl_context = (cl_context) cv::ocl::Context::getDefault().ptr();
    c_queue = xm::ocl::create_queue_device(ocl_context, device_id, true, false);
}

std::map<std::string, xm::ocl::Image2D> xm::DummyCamera::captureWithName() {
    std::map<std::string, xm::ocl::Image2D> map;
    for (auto & prop : properties) {
        cv::Mat cpy;
        src.copyTo(cpy);
        map[prop.name] = xm::ocl::iop::from_cv_mat(cpy, c_queue).waitFor().getImage2D();
    }
    return map;
}

std::vector<xm::ocl::Image2D> xm::DummyCamera::capture() {
    std::vector<xm::ocl::Image2D> images;
    images.reserve(properties.size());
    for (int i = 0; i < properties.size(); i++) {
        cv::Mat cpy;
        src.copyTo(cpy);
        images.push_back(xm::ocl::iop::from_cv_mat(cpy, c_queue).waitFor().getImage2D());
    }
    return images;
}

std::vector<xm::ocl::Image2D> xm::DummyCamera::dequeue() {
    return capture();
}

std::map<std::string, xm::ocl::Image2D> xm::DummyCamera::dequeueWithName() {
    return captureWithName();
}

void xm::DummyCamera::enqueue() {}

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
