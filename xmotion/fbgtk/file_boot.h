//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_FILE_BOOT_H
#define XMOTION_FILE_BOOT_H

#include "gtk/simple_image_window.h"
#include "gtk/cam_params_window.h"
#include "data/json_config.h"
#include "../core/camera/stereo_camera.h"
#include "../core/boot/a_updated_boot.h"
#include "../core/algo/i_logic.h"
#include "../core/filter/i_filter.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm {

    class FileBoot : public xm::UpdatedBoot {

        static inline const auto log =
                spdlog::stdout_color_mt("file_boot");

    protected:
        xm::CamParamsWindow *params_window;
        xm::SimpleImageWindow* window;
        xm::data::JsonConfig config;
        std::string project_file;
    public:
        int boostrap(int &argc, char **&argv) override;

        eox::util::DeltaWorker *worker() override;

        void open_project(const char *argv) override;

        ~FileBoot() override;
    };

} // xm

#endif //XMOTION_FILE_BOOT_H
