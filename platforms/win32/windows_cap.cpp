//
// Created by henryco on 4/19/24.
//

#include "../../../xmotion/camera/stereo_camera.h"

int xm::cap::video_capture_api() {
    return 700; //DirectShow
}

int xm::cap::index_from_id(const std::string &id) {
    // TODO DirectShow Implementation
    return 0;
}

xm::cap::camera_controls xm::cap::query_controls(const std::string &id) {
    // TODO DirectShow Implementation
    return xm::cap::camera_controls();
}

xm::cap::camera_controls xm::cap::read(std::istream &input_stream, const std::string &name) {
    // TODO DirectShow Implementation
    return xm::cap::camera_controls();
}

void xm::cap::save(std::ostream &output_stream, const std::string &name, const camera_controls &control) {
    // TODO DirectShow Implementation
}

void xm::cap::set_control_value(const std::string &device_id, uint prop_id, int value) {
    // TODO DirectShow Implementation
}