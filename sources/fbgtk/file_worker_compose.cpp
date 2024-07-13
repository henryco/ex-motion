//
// Created by henryco on 13/07/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/fbgtk/data/json_ocv.h"
#include "../../xmotion/core/utils/epi_util.h"

namespace xm {
    void FileWorker::opt_compose() {
        const auto out_name = config.compose.name;

        std::vector<cv::Mat> rt_pairs;
        rt_pairs.reserve(config.compose.chain.size());
        for (const auto &pair_name: config.compose.chain) {
            const std::filesystem::path root = project_file;
            const std::filesystem::path name = pair_name;
            const std::filesystem::path path = (name.is_absolute() ? name : (root.parent_path() / name));

            auto files = xm::data::list_files(path);
            std::sort(files.begin(), files.end(), xm::data::numeric_comparator_asc);

            log->info("Reading chain-calibration file: {}, {}", pair_name, path.string());
            for (const auto &file: files) {
                log->info("Reading chain-calibration file: {}, {}", pair_name, file);
                rt_pairs.push_back(xm::data::ocv::read_chain_calibration(file).RT);
            }
        }

        if (rt_pairs.size() > 1)
            throw std::invalid_argument("rt_pairs.size() <= 1");


        // actual logic
        const auto origin_rt = xm::util::epi::chain_merge(rt_pairs);
        // actual logic


        std::filesystem::path root = project_file;
        root = root.parent_path();
        const std::filesystem::path name = out_name.ends_with(".json") ? out_name : (out_name + ".json");
        const std::string file = (root / name).string();

        log->info("Saving chain composition results");
        xm::data::ocv::write_chain_calibration(file, {
                .name = out_name,
                .R = origin_rt(cv::Rect(0, 0, 3, 3)),
                .T = origin_rt(cv::Rect(3, 0, 1, 3)),
                .RT = origin_rt
        });
        log->info("Saved: {}", file);

        std::exit(0);
    }
}