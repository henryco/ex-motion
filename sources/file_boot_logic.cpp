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
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = config.camera.capture[0].name;
            const std::string file = (root.parent_path() / name).string();

            log->info("Saving calibration results");

            xm::data::ocv::write_calibration(file, {
                    .name = name.string(),
                    .K = results.K,
                    .D = results.D,
                    .error = results.mre_1
            });
        }
        // TODO MORE TYPES
    }

    void FileBoot::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();
            logic->debug(true);

            auto calib = dynamic_cast<xm::Calibration *>(logic.get());
            calib->init({
                                .delay = config.calibration.delay,
                                .total = config.calibration.total,
                                .columns = config.calibration.pattern.columns,
                                .rows = config.calibration.pattern.rows,
                                .size = config.calibration.pattern.size,
                                .width = config.camera.capture[0].width,
                                .height = config.camera.capture[0].height,
                                .fx = config.calibration.intrinsics[0].f.x,
                                .fy = config.calibration.intrinsics[0].f.y,
                                .cx = config.calibration.intrinsics[0].c.x,
                                .cy = config.calibration.intrinsics[0].c.y,
                                .fix_f = config.calibration.intrinsics[0].f.fix,
                                .fix_c = config.calibration.intrinsics[0].c.fix,
                        });
            return;
        }
        // TODO more types
        throw std::runtime_error("Invalid config type");
    }

}