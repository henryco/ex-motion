//
// Created by henryco on 5/10/24.
//

#include "../../xmotion/core/dnn/pose_pipeline.h"

namespace eox::dnn {

    std::chrono::nanoseconds PosePipeline::timestamp() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
    }

    eox::dnn::box::Model PosePipeline::getDetectorModel() const {
        return detector.get_model_type();
    }

    void PosePipeline::setDetectorModel(eox::dnn::box::Model model) {
        detector.set_model_type(model);
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

    void PosePipeline::setRoiScale(float scale) {
        roi_scale = scale;
    }

    void PosePipeline::setRoiPaddingX(float padding) {
        roi_padding_x = padding;
    }

    void PosePipeline::setRoiPaddingY(float padding) {
        roi_padding_y = padding;
    }

    void PosePipeline::setRoiClampWindow(float window) {
        roi_clamp_window = window;
    }

    void PosePipeline::setRoiPredictionWindow(float window) {
        roi_center_window = window;
    }

    void PosePipeline::setRoiRollbackWindow(float window) {
        roi_rollback_window = window;
    }

    void PosePipeline::setRoiMargin(float margin) {
        roi_margin = margin;
    }

    void PosePipeline::setRoiThreshold(float threshold) {
        threshold_roi = threshold;
    }

    float PosePipeline::getRoiThreshold() const {
        return threshold_roi;
    }

    float PosePipeline::getRoiRollbackWindow() const {
        return roi_rollback_window;
    }

    float PosePipeline::getRoiScale() const {
        return roi_scale;
    }

    float PosePipeline::getRoiMargin() const {
        return roi_margin;
    }

    float PosePipeline::getRoiPaddingX() const {
        return roi_padding_x;
    }

    float PosePipeline::getRoiPaddingY() const {
        return roi_padding_y;
    }

    float PosePipeline::getRoiClampWindow() const {
        return roi_clamp_window;
    }

    float PosePipeline::getRoiPredictionWindow() const {
        return roi_center_window;
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

    void PosePipeline::setMarksThreshold(float threshold) {
        threshold_marks = threshold;
    }

    float PosePipeline::getMarksThreshold() const {
        return threshold_marks;
    }

    void PosePipeline::enableSegmentation(bool enable) {
        pose.set_segmentation(enable);
    }

    bool PosePipeline::segmentation() const {
        return pose.segmentation();
    }

} // eox