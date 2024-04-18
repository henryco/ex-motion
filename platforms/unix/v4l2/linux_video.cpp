//
// Created by henryco on 11/25/23.
//

#include "linux_video.h"
#include <linux/videodev2.h>
#include <fcntl.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <map>

std::vector<eox::v4l2::V4L2_QueryCtrl> eox::v4l2::get_camera_props(const std::string &device) {
    std::vector<eox::v4l2::V4L2_QueryCtrl> properties;

    const int file_descriptor = open(device.c_str(), O_RDWR);

    if (file_descriptor == -1) {
        std::cerr << "Cannot open video device: " << device << std::endl;
        return {};
    }

    try {

        eox::v4l2::V4L2_QueryCtrl queryctrl;
        queryctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;

        while (0 == ioctl(file_descriptor, VIDIOC_QUERYCTRL, &queryctrl)) {
            if (!(queryctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {

                if (queryctrl.type != V4L2_CTRL_TYPE_CTRL_CLASS) {
                    eox::v4l2::V4L2_Control ctr = {.id = queryctrl.id};
                    if (ioctl(file_descriptor, VIDIOC_G_CTRL, &ctr) == -1) {
                        std::cerr << "Cannot retrieve value for control: "
                                  << ctr.id << " [" << device << "]"
                                  << std::endl;
                    }
                    queryctrl.value = ctr.value;
                }

                // TODO: query menu items name

                properties.push_back(queryctrl);
            }

            queryctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
        }

        if (errno != EINVAL) {
            std::cerr << "fails for reasons other than reaching the end of the control list" << std::endl;
            close(file_descriptor);
            return {};
        }

    } catch (...) {
        close(file_descriptor);
        throw;
    }

    close(file_descriptor);
    return properties;
}

bool eox::v4l2::set_camera_prop(const std::string &device_id, uint prop_id, int prop_value) {
    // not the most efficient way to pass data, but it's ok
    return set_camera_prop(device_id, {
            .id = prop_id,
            .value = prop_value
    });
}

bool eox::v4l2::set_camera_prop(const std::string &device_id, eox::v4l2::V4L2_Control control) {
    // not the most efficient way to pass data, but it's ok.
    return set_camera_prop(
            device_id,
            std::vector<eox::v4l2::V4L2_Control>({control})
    )[0];
}

std::vector<bool> eox::v4l2::set_camera_prop(const std::string &device, std::vector<eox::v4l2::V4L2_Control> controls) {
    const int file_descriptor = open(device.c_str(), O_RDWR);

    std::vector<bool> results(controls.size());

    if (file_descriptor == -1) {
        std::cerr << "Cannot open video device: " << device << std::endl;
        return results;
    }

    try {

        for (int i = 0; i < controls.size(); ++i) {
            auto &control = controls[i];
            if (ioctl(file_descriptor, VIDIOC_S_CTRL, &control) == -1) {
                std::cerr << "Cannot set control value: " << control.id << std::endl;
                results[i] = false;
            } else {
                results[i] = true;
            }
        }

    } catch (...) {
        close(file_descriptor);
        throw;
    }

    close(file_descriptor);
    return results;
}

void eox::v4l2::reset_defaults(const std::string &device_id) {
    const auto props = get_camera_props(device_id);
    std::vector<eox::v4l2::V4L2_Control> controls;

    for (const auto &prop: props) {
        if (prop.type == 6)
            continue;
        controls.push_back({
                                   .id = prop.id,
                                   .value = prop.default_value
                           });
    }

    set_camera_prop(device_id, controls);
}

void eox::v4l2::write_control(std::ostream &os, const eox::v4l2::V4L2_Control &control) {
    const eox::v4l2::serial_v4l2_control serial = { .id = control.id, .value = control.value };
    const auto data = reinterpret_cast<const char *>(&serial);
    os.write(data, sizeof(eox::v4l2::serial_v4l2_control));
}

void eox::v4l2::write_control(std::ostream &os, const std::string &device_id, const std::vector<eox::v4l2::V4L2_Control>& controls) {
    const uint32_t header[] = {
            (uint32_t) controls.size(),
            (uint32_t) device_id.length(),
    };

    os.write(reinterpret_cast<const char *>(header), sizeof(header));
    os.write(device_id.c_str(), (uint32_t) device_id.length());

    for (const auto &control: controls) {
        eox::v4l2::write_control(os, control);
    }
}

eox::v4l2::V4L2_Control eox::v4l2::read_control(std::istream &is) {
    eox::v4l2::serial_v4l2_control data;
    is.read(reinterpret_cast<char *>(&data), sizeof(eox::v4l2::serial_v4l2_control));
    return {
        .id = data.id,
        .value = data.value
    };
}

std::vector<eox::v4l2::V4L2_Control> eox::v4l2::read_control(std::istream &is, size_t num) {
    std::vector<eox::v4l2::V4L2_Control> vec;
    vec.reserve(num);
    for (int i = 0; i < num; i++) {
        vec.push_back(eox::v4l2::read_control(is));
    }
    return vec;
}

std::map<std::string, std::vector<eox::v4l2::V4L2_Control>> eox::v4l2::read_controls(std::istream &is) {
    std::map<std::string, std::vector<eox::v4l2::V4L2_Control>> map;

    while (is.peek() != EOF) {
        uint32_t header[2];
        is.read(reinterpret_cast<char *>(header), sizeof(header));

        const size_t total_controls = header[0];
        const size_t device_id_len = header[1];

        char *c_str = new char[device_id_len];
        is.read(c_str, (long) device_id_len);

        map.emplace(std::string(c_str), eox::v4l2::read_control(is, total_controls));
    }

    return map;
}