//
// Created by henryco on 4/22/24.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/algo/calibration.h"
#include "../xmotion/data/json_ocv.h"
#include "../xmotion/algo/cross.h"
#include "../xmotion/algo/pose.h"

namespace xm {

    void FileBoot::update(float delta, float _, float fps) {
        window->setFps((int) fps);

        logic->proceed(delta, camera->capture());
        process_results();

        if (!bypass)
            window->refresh(logic->frames());
        else
            window->refresh(false);
    }

    void FileBoot::process_results() {
        if (config.type == data::CALIBRATION) {
            const auto results = (static_cast<xm::Calibration *>(logic.get()))->result();
            if (!results.ready)
                return;

            logic->stop();

            const std::filesystem::path root = project_file;
            const std::filesystem::path name = config.calibration.name + ".json";
            const std::string file = (root.parent_path() / name).string();

            log->info("Saving calibration results");

            xm::data::ocv::write_calibration(file, {
                    .name = config.calibration.name,
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

            log->info("saved");
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {

            const auto results = (static_cast<xm::CrossCalibration *>(logic.get()))->result();
            if (!results.ready)
                return;

            logic->stop();

            const auto total = results.calibrated.size();
            for (int i = 0; i < total; ++i) {
                const std::string postfix = (total == 1) ? "" : ("_" + std::to_string(i));

                const std::filesystem::path root = project_file;
                const std::filesystem::path name = config.calibration.name + postfix + ".json";
                const std::string file = (root.parent_path() / name).string();

                log->info("Saving cross calibration results: [{}]", i);
                const auto pair = results.calibrated.at(i);
                xm::data::ocv::write_cross_calibration(file, {
                    .name = config.calibration.name + postfix,
                    .R = pair.R,
                    .T = pair.T,
                    .E = pair.E,
                    .F = pair.F,
                    .RTp = pair.RTp,
                    .RT0 = pair.RT0,
                    .error = pair.mre
                });

                log->info("saved: [{}]", i);
            }

            return;
        }

        if (config.type == data::POSE) {
            // TODO

            return;
        }
    }

    void FileBoot::prepare_logic() {
        if (config.type == data::CALIBRATION) {
            logic = std::make_unique<xm::Calibration>();
            logic->debug(config.misc.debug);

            const auto w = config.camera.capture[0].width;
            const auto h = config.camera.capture[0].height;
            const auto r = config.camera.capture[0].rotate;

            xm::calib::Initial params = {
                    .delay = config.calibration.delay,
                    .total = config.calibration.total,
                    .columns = config.calibration.pattern.columns,
                    .rows = config.calibration.pattern.rows,
                    .size = config.calibration.pattern.size,
                    .sb = config.calibration.pattern.type == xm::data::board::Type::RADON,
                    .width = r ? h : w,
                    .height = r ? w : h
            };

            params.fx = config.calibration.intrinsics.f.x;
            params.fy = config.calibration.intrinsics.f.y;
            params.cx = config.calibration.intrinsics.c.x;
            params.cy = config.calibration.intrinsics.c.y;
            params.fix_f = config.calibration.intrinsics.f.fix;
            params.fix_c = config.calibration.intrinsics.c.fix;

            (static_cast<xm::Calibration *>(logic.get()))->init(params);
            return;
        }

        if (config.type == data::CROSS_CALIBRATION) {
            logic = std::make_unique<xm::CrossCalibration>();
            logic->debug(config.misc.debug);

            xm::cross::Initial params = {
                    .delay = config.calibration.delay,
                    .total = config.calibration.total,
                    .columns = config.calibration.pattern.columns,
                    .rows = config.calibration.pattern.rows,
                    .size = config.calibration.pattern.size,
                    .sb = config.calibration.pattern.type == xm::data::board::Type::RADON
            };

            params.closed = config.calibration.cross.closed;
            params.views = (int) config.calibration.cross.calibrated.size();
            if (params.views <= 0)
                throw std::runtime_error("Cross calibration requires at least two calibrated cameras");

            std::vector<cv::Mat> K, D;
            for (const auto &item: config.calibration.cross.calibrated) {
                const std::filesystem::path root = project_file;
                const std::filesystem::path name = item + ".json";
                const std::string file = (root.parent_path() / name).string();

                const auto calibration = xm::data::ocv::read_calibration(file);
                log->info("Read calibration file: {}", calibration.name);
                K.push_back(calibration.K);
                D.push_back(calibration.D);
            }

            params.K = K;
            params.D = D;

            (static_cast<xm::CrossCalibration *>(logic.get()))->init(params);
            return;
        }

        if (config.type == data::POSE) {
            logic = std::make_unique<xm::Pose>();
            logic->debug(config.misc.debug);

            const xm::nview::Initial params = {
                    .detector_model = static_cast<xm::nview::DetectorModel>(static_cast<int>(config.pose.detector)),
                    .body_model = static_cast<xm::nview::BodyModel>(static_cast<int>(config.pose.body)),
                    .roi_center_window = config.pose.roi.center_window,
                    .roi_clamp_window = config.pose.roi.clamp_window,
                    .roi_margin = config.pose.roi.margin,
                    .roi_scale = config.pose.roi.scale,
                    .roi_padding_x = config.pose.roi.padding_x,
                    .roi_padding_y = config.pose.roi.padding_y,
                    .threshold_detector = config.pose.threshold.detector,
                    .threshold_presence = config.pose.threshold.presence,
                    .threshold_pose = config.pose.threshold.presence,
                    .filter_velocity_factor = config.pose.filter.velocity,
                    .filter_windows_size = config.pose.filter.window,
                    .filter_target_fps = config.pose.filter.fps,
                    .segmentation = config.pose.segmentation,
                    .threads = config.pose.threads <= 0
                            ? config.misc.cpu
                            : std::min(config.pose.threads, config.misc.cpu),
                    .views = (int) config.camera.capture.size()
            };

            (static_cast<xm::Pose *>(logic.get()))->init(params);
            return;
        }

        throw std::runtime_error("Invalid config type");
    }

}
#pragma clang diagnostic pop