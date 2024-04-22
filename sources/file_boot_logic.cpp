//
// Created by henryco on 4/22/24.
//

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/algo/calibration.h"

namespace xm {

    void FileBoot::update(float delta, float _, float fps) {
        logic->proceed(delta, camera.capture());

        // TODO RESULT PROCESSING

        window->setFps((int) fps);

        if (!bypass)
            window->refresh(logic->frames());
        else
            window->refresh(false);
    }

    void FileBoot::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();
            logic->stop();
            // TODO: init
            return;
        }
        // TODO more types
        throw std::runtime_error("Invalid config type");
    }

}