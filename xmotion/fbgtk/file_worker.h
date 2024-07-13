//
// Created by henryco on 13/06/24.
//

#ifndef XMOTION_FILE_WORKER_H
#define XMOTION_FILE_WORKER_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "gtk/simple_image_window.h"
#include "gtk/cam_params_window.h"
#include "data/json_config.h"

#include "../core/utils/delta_loop.h"
#include "../core/algo/i_logic.h"
#include "../core/filter/i_filter.h"
#include "../core/utils/thread_pool.h"
#include "../core/camera/stereo_camera.h"

namespace xm {

    class FileWorker : public eox::util::DeltaWorker {
        static inline const auto log =
                spdlog::stdout_color_mt("file_worker");

    protected:
        // ==== pointers managed externally ====
        xm::CamParamsWindow *params_window;
        xm::SimpleImageWindow *window;
        // ==== pointers managed externally ====

        sigc::connection idle_connection;

        std::vector<std::vector<std::unique_ptr<xm::Filter>>> filters;
        std::unique_ptr<xm::StereoCamera> camera;
        std::unique_ptr<xm::Logic> logic;

        xm::data::JsonConfig config;
        std::string project_file;

        bool do_filter = false;
        bool bypass = false;

    public:
        FileWorker(xm::SimpleImageWindow *_window,
                   xm::CamParamsWindow *params_window,
                   const xm::data::JsonConfig &_config,
                   const std::string &_project_file);

        void update(float dt, float latency, float fps) override;

    private:
        void filter_frames(std::vector<xm::ocl::Image2D> &frames_in_out);

        void prepare_gui();

        void prepare_cam();

        void prepare_logic();

        void prepare_filters();

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

        void opt_compose();

        void on_single_results();

        void on_chain_results();

        void on_cross_results();

        void on_pose_results();

        void update_gui(float fps);
    };

} // xm

#endif //XMOTION_FILE_WORKER_H
