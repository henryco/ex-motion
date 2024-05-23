//
// Created by henryco on 4/22/24.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

#include "../../xmotion/fbgtk/file_boot.h"
#include "../../xmotion/core/algo/calibration.h"
#include "../../xmotion/fbgtk/data/json_ocv.h"
#include "../../xmotion/core/algo/chain.h"
#include "../../xmotion/core/algo/pose.h"

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

    void FileBoot::prepare_logic() {
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

        throw std::runtime_error("Invalid config type");
    }


    void FileBoot::opt_single_calibration() {
        logic = std::make_unique<xm::Calibration>();
        logic->debug(config.misc.debug);

        const auto w = config.camera.capture[0].region.w;
        const auto h = config.camera.capture[0].region.h;
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
    }

    void FileBoot::opt_chain_calibration() {
        logic = std::make_unique<xm::ChainCalibration>();
        logic->debug(config.misc.debug);

        xm::chain::Initial params = {
                .delay = config.calibration.delay,
                .total = config.calibration.total,
                .columns = config.calibration.pattern.columns,
                .rows = config.calibration.pattern.rows,
                .size = config.calibration.pattern.size,
                .sb = config.calibration.pattern.type == xm::data::board::Type::RADON
        };

        params.closed = config.calibration.chain.closed;
        params.views = (int) config.calibration.chain.intrinsics.size();
        if (params.views <= 0)
            throw std::runtime_error("Cross calibration requires at least two calibrated cameras");

        std::vector<cv::Mat> K, D;
        for (const auto &item: config.calibration.chain.intrinsics) {
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = item;
            const std::string file = (name.is_absolute() ? name : (root.parent_path() / name)).string();

            const auto calibration = xm::data::ocv::read_calibration(file);
            log->info("Reading calibration file: {}, {}", calibration.name, file);
            K.push_back(calibration.K);
            D.push_back(calibration.D);
        }

        params.K = K;
        params.D = D;

        (static_cast<xm::ChainCalibration *>(logic.get()))->init(params);
    }

    void FileBoot::opt_cross_calibration() {
        // TODO
    }

    void FileBoot::opt_pose_estimation() {
        logic = std::make_unique<xm::Pose>();
        logic->debug(config.misc.debug);

        std::vector<xm::nview::Device> vec;
        vec.reserve(config.pose.devices.size());
        int i = 0;
        for (const auto &device: config.pose.devices) {
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = device.calibration;
            const std::string file = (name.is_absolute() ? name : (root.parent_path() / name)).string();

            log->info("Reading calibration file: {}, {}", device.calibration, file);
            const auto calibration = xm::data::ocv::read_calibration(file);
            const auto rotate = config.camera.capture[i].rotate;
            const auto width = config.camera.capture[i].region.w;
            const auto height = config.camera.capture[i].region.h;
            vec.push_back(
                    {
                            .detector_model = static_cast<xm::nview::DetectorModel>(static_cast<int>(device.model.detector)),
                            .body_model = static_cast<xm::nview::BodyModel>(static_cast<int>(device.model.body)),
                            .roi_rollback_window = device.roi.rollback_window,
                            .roi_center_window = device.roi.center_window,
                            .roi_clamp_window = device.roi.clamp_window,
                            .roi_margin = device.roi.margin,
                            .roi_scale = device.roi.scale,
                            .roi_padding_x = device.roi.padding_x,
                            .roi_padding_y = device.roi.padding_y,
                            .threshold_detector = device.threshold.detector,
                            .threshold_marks = device.threshold.marks,
                            .threshold_pose = device.threshold.pose,
                            .threshold_roi = device.threshold.roi,
                            .filter_velocity_factor = device.filter.velocity,
                            .filter_windows_size = device.filter.window,
                            .filter_target_fps = device.filter.fps,
                            .undistort_source = device.undistort.source,
                            .undistort_points = device.undistort.points,
                            .undistort_alpha = device.undistort.alpha,
                            .width = rotate ? height : width,
                            .height = rotate ? width : height,
                            .K = calibration.K,
                            .D = calibration.D
                    });

            i++;
        }

        std::vector<xm::util::epi::CalibPair> pairs;
        pairs.reserve(config.pose.stereo.size());
        for (const auto &pair_name: config.pose.stereo) {
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = pair_name;
            const std::string file = (name.is_absolute() ? name : (root.parent_path() / name)).string();

            log->info("Reading cross-calibration file: {}, {}", pair_name, file);
            const auto cross_calibration = xm::data::ocv::read_cross_calibration(file);
//                pairs.push_back({
//                    .E = cross_calibration.E,
//                    .F = cross_calibration.F,
//                    .RT = cross_calibration.RT,
//                    .RTo = cross_calibration.RTo
//                });
            // TODO
        }

        const xm::nview::Initial params = {
                .devices = vec,
                .epi_matrix = xm::util::epi::Matrix::from_chain(pairs), // TODO
                .segmentation = config.pose.segmentation,
                .show_epilines = config.pose.show_epilines,
                .threads = config.pose.threads <= 0
                           ? config.misc.cpu
                           : std::min(config.pose.threads, config.misc.cpu),
        };

        (static_cast<xm::Pose *>(logic.get()))->init(params);
    }

    void FileBoot::on_single_results() {
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
    }

    void FileBoot::on_chain_results() {
        const auto results = (static_cast<xm::ChainCalibration *>(logic.get()))->result();
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
                    .RT = pair.RT,
                    .error = pair.mre
            });

            log->info("saved: [{}]", i);
        }
    }

    void FileBoot::on_cross_results() {
        // TODO
    }

    void FileBoot::on_pose_results() {
        // TODO
    }
}
#pragma clang diagnostic pop