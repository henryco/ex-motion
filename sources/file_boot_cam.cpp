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

            std::filesystem::path file = c.name + ".xcam";
            std::filesystem::path conf = project_dir / file;

            if (!std::filesystem::exists(conf)) {
                log->debug("No configuration file found for camera device: {} | {} ", c.id, c.name);
                continue;
            }

            std::ifstream is(conf);
            log->debug("reading configuration for camera device: {} | {}", c.id, c.name);
            camera.read(is, c.id, c.name);
            is.close();
        }
    }

}