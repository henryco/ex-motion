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

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm {

    class FileBoot : public xm::UpdatedBoot {

        static inline const auto log =
                spdlog::stdout_color_mt("file_boot");

    protected:
        std::unique_ptr<xm::CamParamsWindow> params_window;
        std::unique_ptr<xm::SimpleImageWindow> window;
        std::unique_ptr<xm::StereoCamera> camera;
        std::unique_ptr<xm::Logic> logic;

        xm::data::JsonConfig config;
        std::string project_file;

        bool bypass = false;

    public:
        int boostrap(int &argc, char **&argv) override;

        void update(float delta, float latency, float fps) override;

        void open_project(const char *argv) override;

    private:
        void prepare_gui();

        void prepare_cam();

        void prepare_logic();

        void load_device_params();

        void process_results();

        int on_camera_update(const std::string &device_id, uint id, int value);

        void on_camera_reset(const std::string &device_id);

        void on_camera_save(const std::string &device_id);

        void on_camera_read(const std::string &device_id, const std::string &name);

        void opt_single_calibration();

        void opt_chain_calibration();

        void opt_cross_calibration();

        void opt_pose_estimation();

        void on_single_results();

        void on_chain_results();

        void on_cross_results();

        void on_pose_results();
    };

} // xm

#endif //XMOTION_FILE_BOOT_H
