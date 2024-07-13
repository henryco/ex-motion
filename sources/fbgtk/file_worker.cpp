//
// Created by henryco on 13/06/24.
//

#include "../../xmotion/fbgtk/file_worker.h"

namespace xm {

    FileWorker::FileWorker(xm::SimpleImageWindow *_window,
                           xm::CamParamsWindow *_params_window,
                           const xm::data::JsonConfig &_config,
                           const std::string &_project_file):
            config(_config), project_file(_project_file),
            window(_window), params_window(_params_window) {

        prepare_filters();
        prepare_logic();
        prepare_cam();
        prepare_gui();
    }

    void xm::FileWorker::update(float dt, float latency, float fps) {
//        const auto t0 = std::chrono::system_clock::now();

        std::vector<xm::ocl::Image2D> frames = camera->dequeue();
        camera->enqueue();

        filter_frames(frames);
        logic->proceed(dt, frames);
        process_results();

        if (!bypass)
            update_gui(fps);


//        const auto t1 = std::chrono::system_clock::now();
//        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
//        log->info("time: {}", d);
    }


    void FileWorker::process_results() {
        if (config.type == data::CALIBRATION) {
            on_single_results();
            return;
        }

        if (config.type == data::CHAIN_CALIBRATION) {
            on_chain_results();
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {
            on_cross_results();
            return;
        }

        if (config.type == data::POSE) {
            on_pose_results();
            return;
        }

        throw std::runtime_error("Invalid config type");
    }

    void FileWorker::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            opt_single_calibration();
            return;
        }

        if (config.type == data::CHAIN_CALIBRATION) {
            opt_chain_calibration();
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {
            opt_cross_calibration();
            return;
        }

        if (config.type == data::POSE) {
            opt_pose_estimation();
            return;
        }

        if (config.type == data::COMPOSE) {
            opt_compose();
        }

        throw std::runtime_error("Invalid config type");
    }


} // xm