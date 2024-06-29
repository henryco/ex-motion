//
// Created by henryco on 4/21/24.
//

#include <fstream>
#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/core/camera/d_dummy_camera.h"

namespace xm {

    void FileWorker::prepare_cam() {
        camera = config.misc.capture_dummy
                ? std::make_unique<xm::DummyCamera>()
                : std::make_unique<xm::StereoCamera>();
        camera->setFastMode(config.misc.capture_fast);
        for (const auto &c: config.captures) {
            camera->open({
                                .device_id = c.id,
                                .name = c.name,
                                .codec = c.codec,
                                .width = c.width,
                                .height = c.height,
                                .fps = c.fps,
                                .buffer = c.buffer,
                                .flip_x = c.flip.x,
                                .flip_y = c.flip.y,
                                .rotate = c.rotate,
                                .x = c.region.x,
                                .y = c.region.y,
                                .w = c.region.w,
                                .h = c.region.h,
                        });
            on_camera_read(c.id, c.name);
        }
    }

    void FileWorker::on_camera_save(const std::string &device_id) {
        const std::filesystem::path project_dir = xm::data::prepare_project_dir(project_file);
        const std::filesystem::path parent = xm::data::create_dir_rec(project_dir / "cam");
        for (const auto &c: config.captures) {
            if (c.id != device_id)
                continue;

            std::filesystem::path conf = parent / (c.name + ".xcam");
            std::ofstream os(conf);
            camera->save(os, device_id, c.name);
            os.close();

            log->info("saved camera settings for: [{}|{}], {}", device_id, c.name, conf.string());
            return;
        }
    }

    void FileWorker::on_camera_read(const std::string &device_id, const std::string &name) {
        const std::filesystem::path project_dir = xm::data::prepare_project_dir(project_file);
        const std::filesystem::path conf = project_dir / "cam" / (name + ".xcam");

        if (!std::filesystem::exists(conf)) {
            log->debug("No configuration file found for camera device: [{}|{}], {}", device_id, name, conf.string());
            return;
        }

        std::ifstream is(conf);
        log->info("reading configuration for camera device: [{}|{}], {}", device_id, name, conf.string());
        camera->read(is, device_id, name);
        is.close();
    }

    int FileWorker::on_camera_update(const std::string &device_id, uint id, int value) {
        camera->setControl(device_id, id, value);
        log->debug("updated camera settings for: {} | {}:{}", device_id, id, value);
        return value;
    }

    void FileWorker::on_camera_reset(const std::string &device_id) {
        camera->resetControls(device_id);
        log->debug("reset camera settings for: {}", device_id);
    }

}