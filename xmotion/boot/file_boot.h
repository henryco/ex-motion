//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_FILE_BOOT_H
#define XMOTION_FILE_BOOT_H

#include "../gtk/simple_image_window.h"
#include "../gtk/cam_params_window.h"
#include "../data/json_config.h"
#include "../camera/stereo_camera.h"
#include "updated_boot.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm {

    class FileBoot : public xm::UpdatedBoot {

        static inline const auto log =
                spdlog::stdout_color_mt("file_boot");

    protected:
        std::unique_ptr<xm::CamParamsWindow> params_window;
        std::unique_ptr<xm::SimpleImageWindow> window;

        xm::data::JsonConfig config;
        std::string project_path;
        xm::StereoCamera camera;

    public:
        int boostrap(int &argc, char **&argv) override;

        void update(float delta, float latency, float fps) override;

        void open_project(const char *argv) override;

    private:
        void prepare_gui();

        void prepare_cam();

    };

} // xm

#endif //XMOTION_FILE_BOOT_H
