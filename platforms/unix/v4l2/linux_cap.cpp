//
// Created by henryco on 4/19/24.
// Linux specific camera functions
//

#include "../../../xmotion/camera/stereo_camera.h"
#include "linux_video.h"

int xm::cap::video_capture_api() {
    return 200; //V4L2
}

int xm::cap::index_from_id(const std::string &id) {
    // expects: /dev/videoX
    // returns: X
    if (id.length() < 10)
        throw std::runtime_error("invalid device id: " + id);
    return std::stoi(id.substr(10));
}

xm::cap::camera_controls xm::cap::query_controls(const std::string &id) {
    const auto control = eox::v4l2::get_camera_props(id);
    std::vector<camera_control> controls;

    for (const auto &ctr: control) {
        if (ctr.type == 6)
            continue;
        controls.push_back({
            .id = ctr.id,
//            .type = ctr.type,
            .name = std::string(reinterpret_cast<const char *>(ctr.name), 32),
            .min = ctr.minimum,
            .max = ctr.maximum,
            .step = ctr.step,
            .default_value = ctr.default_value,
            .value = ctr.value
        });
    }

    return {.id = id,.controls = controls};
}

void xm::cap::set_control_value(const std::string &device_id, uint prop_id, int value) {
    eox::v4l2::set_camera_prop(device_id, prop_id, value);
}

void xm::cap::save(std::ostream &output_stream, const std::string &name, const camera_controls &control) {
    int32_t backend_header[1] = {(int32_t) video_capture_api()};
    output_stream.write(reinterpret_cast<const char *>(backend_header), sizeof(backend_header));

    std::vector<eox::v4l2::V4L2_Control> vec;
    vec.reserve(control.controls.size());
    for (const auto &c: control.controls)
        vec.push_back({.id = c.id, .value = c.value});

    eox::v4l2::write_control(output_stream, name, vec);
}

xm::cap::camera_controls xm::cap::read(std::istream &input_stream, const std::string &name) {
    if (input_stream.peek() == EOF)
        throw std::runtime_error("Cannot read camera settings from file, file is empty [" + name + "]");

    int32_t backend_header[1];
    input_stream.read(reinterpret_cast<char *>(backend_header), sizeof(backend_header));

    if (backend_header[0] != ((int32_t) video_capture_api()))
        throw std::runtime_error("Cannot read camera settings, invalid capture API");

    const auto map = eox::v4l2::read_controls(input_stream);
    if (!map.contains(name))
        throw std::runtime_error("Cannot locate settings in a file for capture: " + name);

    const auto& settings = map.at(name);
    std::vector<camera_control> controls;
    controls.reserve(settings.size());
    for (const auto &s: settings)
        controls.push_back({.id = s.id, .value = s.value});

    return {.id = name,.controls = controls};
}
