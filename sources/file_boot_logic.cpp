//
// Created by henryco on 4/22/24.
//

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/algo/calibration.h"
#include "../xmotion/data/json_ocv.h"

namespace xm {

    void FileBoot::update(float delta, float _, float fps) {
        window->setFps((int) fps);

        logic->proceed(delta, camera.capture());
        process_results();

//        std::this_thread::sleep_for(std::chrono::milliseconds(25));

        if (!bypass)
            window->refresh(logic->frames());
        else
            window->refresh(false);
    }

    void FileBoot::process_results() {
        if (config.type == data::CALIBRATION) {
            const auto results = (dynamic_cast<xm::Calibration *>(logic.get()))->result();
            if (!results.ready)
                return;

            logic->stop();

            const std::filesystem::path root = project_file;
            const std::filesystem::path name = config.calibration.name + ".json";
            const std::string file = (root.parent_path() / name).string();

            log->info("Saving calibration results");

            xm::data::ocv::write_calibration(file, {
                    .name = config.camera.capture[0].name,
                    .K = results.K,
                    .D = results.D,
                    .width = results.width,
                    .height = results.height,
                    .fov_x = results.fov_x,
                    .fov_y = results.fov_y,
                    .f = results.f,
                    .c_x = results.c_x,
                    .c_y = results.c_y,
                    .r = results.r,
                    .error = results.mre_1
            });
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {

            return;
        }

        // TODO MORE TYPES
    }

    void FileBoot::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();
            logic->debug(true);

            const auto w = config.camera.capture[0].width;
            const auto h = config.camera.capture[0].height;
            const auto r = config.camera.capture[0].rotate;

            xm::calib::Initial params = {
                    .delay = config.calibration.delay,
                    .total = config.calibration.total,
                    .columns = config.calibration.pattern.columns,
                    .rows = config.calibration.pattern.rows,
                    .size = config.calibration.pattern.size,
                    .width = r ? h : w,
                    .height = r ? w : h
            };

            params.fx = config.calibration.intrinsics.f.x;
            params.fy = config.calibration.intrinsics.f.y;
            params.cx = config.calibration.intrinsics.c.x;
            params.cy = config.calibration.intrinsics.c.y;
            params.fix_f = config.calibration.intrinsics.f.fix;
            params.fix_c = config.calibration.intrinsics.c.fix;

            (dynamic_cast<xm::Calibration *>(logic.get()))->init(params);
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();
            logic->debug(true);

            return;
        }

        // TODO more types
        throw std::runtime_error("Invalid config type");
    }

}