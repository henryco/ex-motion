//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_FILE_BOOT_H
#define XMOTION_FILE_BOOT_H

#include "boot.h"
#include "../utils/delta_loop.h"
#include "../gtk/simple_image_window.h"
#include "../data/json_config.h"
#include "../camera/stereo_camera.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm {

    class FileBoot : public xm::Boot {

        static inline const auto log =
                spdlog::stdout_color_mt("file_boot");

    protected:
        std::unique_ptr<xm::SimpleImageWindow> window;
        eox::util::DeltaLoop deltaLoop;
        xm::data::JsonConfig config;
        xm::StereoCamera camera;
        std::string project_path;

    public:
        int boot(int &argc, char **&argv) override;

        void open_project(const char *argv) override;

        void update(float delta, float latency, float fps);

    private:
        void prepare_gui();
        void prepare_loop();
        void prepare_ocv();
        void prepare_cam();
    };

} // xm

#endif //XMOTION_FILE_BOOT_H
