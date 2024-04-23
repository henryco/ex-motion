//
// Created by henryco on 4/22/24.
//

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/algo/calibration.h"

namespace xm {

    void FileBoot::update(float delta, float _, float fps) {
        window->setFps((int) fps);

        logic->proceed(delta, camera.capture());
        process_results();

        if (!bypass)
            window->refresh(logic->frames());
        else
            window->refresh(false);
    }

    void FileBoot::process_results() {

    }

    void FileBoot::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();

            auto calib = dynamic_cast<xm::Calibration *>(logic.get());
            calib->init({
                                .delay = config.calibration.delay,
                                .total = config.calibration.total,
                                .columns = config.calibration.pattern.columns,
                                .rows = config.calibration.pattern.rows,
                                .size = config.calibration.pattern.size,
                                .width = config.camera.capture[0].width,
                                .height = config.camera.capture[0].height,
                                .fx = config.calibration.intrinsics[0].f_x,
                                .fy = config.calibration.intrinsics[0].f_y,
                                .cx = config.calibration.intrinsics[0].c_x,
                                .cy = config.calibration.intrinsics[0].c_y,
                                .fix = config.calibration.intrinsics[0].fix
                        });
            return;
        }
        // TODO more types
        throw std::runtime_error("Invalid config type");
    }

}