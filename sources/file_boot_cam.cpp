//
// Created by henryco on 4/21/24.
//

#include <fstream>
#include "../xmotion/boot/file_boot.h"

namespace xm {

    void FileBoot::prepare_cam() {
        const auto project_dir = xm::data::prepare_project_dir(project_path);
        camera.setFastMode(config.camera.fast);
        for (const auto &c: config.camera.capture) {
            camera.open(c.id, c.codec, c.width, c.height, c.fps, c.buffer);
            on_camera_read(c.id, c.name);
        }
    }

    void FileBoot::on_camera_save(const std::string &device_id) {
        const auto project_dir = xm::data::prepare_project_dir(project_path);
        for (const auto &c: config.camera.capture) {
            if (c.id != device_id)
                continue;

            std::filesystem::path file = c.name + ".xcam";
            std::filesystem::path conf = project_dir / file;

            std::ofstream os(conf);
            camera.save(os, device_id, c.name);
            os.close();
            return;
        }
    }

    void FileBoot::on_camera_read(const std::string &device_id, const std::string &name) {
        const auto project_dir = xm::data::prepare_project_dir(project_path);
        std::filesystem::path file = name + ".xcam";
        std::filesystem::path conf = project_dir / file;

        if (!std::filesystem::exists(conf)) {
            log->debug("No configuration file found for camera device: {} | {} ", device_id, name);
            return;
        }

        std::ifstream is(conf);
        log->debug("reading configuration for camera device: {} | {}", device_id, name);
        camera.read(is, device_id, name);
        is.close();
    }

    int FileBoot::on_camera_update(const std::string &device_id, uint id, int value) {
        camera.setControl(device_id, id, value);
        return value;
    }

    void FileBoot::on_camera_reset(const std::string &device_id) {
        camera.resetControls(device_id);
    }

}