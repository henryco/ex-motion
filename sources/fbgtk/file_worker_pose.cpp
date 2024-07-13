//
// Created by henryco on 13/07/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/core/algo/pose.h"
#include "../../xmotion/fbgtk/data/json_ocv.h"

namespace xm {

    void FileWorker::on_pose_results() {
        // TODO
    }

    void FileWorker::opt_pose_estimation() {
        logic = std::make_unique<xm::Pose>();
        logic->debug(config.misc.debug);

        std::vector<xm::nview::Device> vec;
        vec.reserve(config.pose.devices.size());
        int i = 0;
        for (const auto &device: config.pose.devices) {
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = device.intrinsics;
            const std::string file = (name.is_absolute() ? name : (root.parent_path() / name)).string();

            log->info("Reading calibration file: {}, {}", device.intrinsics, file);
            const auto calibration = xm::data::ocv::read_calibration(file);
            const auto rotate = config.captures[i].rotate;
            const auto width = config.captures[i].region.w;
            const auto height = config.captures[i].region.h;
            vec.push_back({
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

        xm::util::epi::Matrix epi_matrix;
        if (config.pose.chain._present) {

            int k = 0;
            std::vector<xm::util::epi::CalibPair> pairs;
            pairs.reserve(config.pose.chain.files.size());
            for (const auto &pair_name: config.pose.chain.files) {
                const std::filesystem::path root = project_file;
                const std::filesystem::path name = pair_name;
                const std::filesystem::path path = (name.is_absolute() ? name : (root.parent_path() / name));

                auto files = xm::data::list_files(path);
                std::sort(files.begin(), files.end(), xm::data::numeric_comparator_asc);

                log->info("Reading chain-calibration file: {}, {}", pair_name, path.string());
                for (const auto &file: files) {
                    log->info("Reading chain-calibration file[{}]: {}, {}", k, pair_name, file);
                    const auto chain_calibration = xm::data::ocv::read_chain_calibration(file);
                    pairs.push_back({
                                            .K1 = vec.at(k).K.clone(),
                                            .K2 = vec.at(k + 1).K.clone(),
                                            .RT = chain_calibration.RT,
                                            .E = chain_calibration.E,
                                            .F = chain_calibration.F
                                    });

                    k++;
                    if (k >= config.pose.devices.size())
                        k = 0; // we are spinning
                }
            }

            epi_matrix = xm::util::epi::Matrix::from_chain(pairs, config.pose.chain.closed, false);

        } else if (config.pose.cross._present) {
            // TODO
        }

        log->debug("Epi_matrix: {}", epi_matrix.to_string());

        const xm::nview::Initial params = {
                .devices = vec,
                .epi_matrix = epi_matrix,
                .segmentation = config.pose.segmentation,
                .show_epilines = config.pose.show_epilines,
                .threads = config.pose.threads <= 0
                           ? config.misc.cpu
                           : std::min(config.pose.threads, config.misc.cpu),
        };

        (static_cast<xm::Pose *>(logic.get()))->init(params);
    }

}
