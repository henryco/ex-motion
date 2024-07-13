//
// Created by henryco on 13/07/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/core/algo/calibration.h"
#include "../../xmotion/core/algo/chain.h"
#include "../../xmotion/fbgtk/data/json_ocv.h"

namespace xm {
    void FileWorker::opt_single_calibration() {
        logic = std::make_unique<xm::Calibration>();
        logic->debug(config.misc.debug);

        const auto w = config.captures[0].region.w;
        const auto h = config.captures[0].region.h;
        const auto r = config.captures[0].rotate;

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

    void FileWorker::opt_chain_calibration() {
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

    void FileWorker::opt_cross_calibration() {
        // TODO
    }


    void FileWorker::on_single_results() {
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

    void FileWorker::on_chain_results() {
        const auto results = (static_cast<xm::ChainCalibration *>(logic.get()))->result();
        if (!results.ready)
            return;

        logic->stop();

        const auto total = results.calibrated.size();
        std::filesystem::path root = project_file;
        root = root.parent_path();

        if (total > 1)
            root = xm::data::create_dir_rec(root / config.calibration.name);

        for (int i = 0; i < total; ++i) {
            const std::filesystem::path name = (total == 1 ? config.calibration.name : std::to_string(i)) + ".json";
            const std::string file = (root / name).string();

            log->info("Saving chain calibration results: [{}]", i);
            const auto pair = results.calibrated.at(i);
            xm::data::ocv::write_chain_calibration(file, {
                    .name = config.calibration.name + (total == 1 ? "" : ("_" + std::to_string(i))),
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

    void FileWorker::on_cross_results() {
        // TODO
    }
}