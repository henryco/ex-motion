//
// Created by henryco on 4/19/24.
//

#include "../../agnostic_cap.h"

int platform::cap::video_capture_api() {
    return 700; //DirectShow
}

int platform::cap::index_from_id(const std::string &id) {
    // TODO DirectShow Implementation
    return 0;
}

platform::cap::camera_controls platform::cap::query_controls(const std::string &id) {
    // TODO DirectShow Implementation
    return cap::camera_controls();
}

platform::cap::camera_controls platform::cap::read(std::istream &input_stream, const std::string &name) {
    // TODO DirectShow Implementation
    return cap::camera_controls();
}

void platform::cap::save(std::ostream &output_stream, const std::string &name, const camera_controls &control) {
    // TODO DirectShow Implementation
}

void platform::cap::set_control_value(const std::string &device_id, uint prop_id, int value) {
    // TODO DirectShow Implementation
}