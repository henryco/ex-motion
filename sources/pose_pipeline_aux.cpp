//
// Created by henryco on 5/10/24.
//

#include "../xmotion/dnn/pose_pipeline.h"

namespace eox {

    std::chrono::nanoseconds PosePipeline::timestamp() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
    }

    void PosePipeline::setBodyModel(eox::dnn::pose::Model model) {
        pose.set_model_type(model);
    }

    eox::dnn::pose::Model PosePipeline::bodyModel() const {
        return pose.get_model_type();
    }

    float PosePipeline::getPoseThreshold() const {
        return threshold_pose;
    }

    void PosePipeline::setPoseThreshold(float threshold) {
        threshold_pose = threshold;
    }

    void PosePipeline::setDetectorThreshold(float threshold) {
        threshold_detector = threshold;
    }

    float PosePipeline::getDetectorThreshold() const {
        return threshold_detector;
    }

    void PosePipeline::setFilterWindowSize(int size) {
        f_win_size = size;
        for (auto &filter: filters) {
            filter.setWindowSize(size);
        }
    }

    void PosePipeline::setFilterVelocityScale(float scale) {
        f_v_scale = scale;
        for (auto &filter: filters) {
            filter.setVelocityScale(scale);
        }
    }

    void PosePipeline::setFilterTargetFps(int fps) {
        f_fps = fps;
        for (auto &filter: filters) {
            filter.setTargetFps(fps);
        }
    }

    float PosePipeline::getFilterVelocityScale() const {
        return f_v_scale;
    }

    int PosePipeline::getFilterTargetFps() const {
        return f_fps;
    }

    int PosePipeline::getFilterWindowSize() const {
        return f_win_size;
    }

    void PosePipeline::setPresenceThreshold(float threshold) {
        threshold_presence = threshold;
    }

    float PosePipeline::getPresenceThreshold() const {
        return threshold_presence;
    }

    void PosePipeline::enableSegmentation(bool enable) {
        pose.set_segmentation(enable);
    }

    bool PosePipeline::segmentation() const {
        return pose.segmentation();
    }

} // eox