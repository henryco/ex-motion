//
// Created by henryco on 1/10/24.
//

#include "../xmotion/dnn/pose_pipeline.h"

namespace eox::dnn {

    void PosePipeline::init() {
        filters.clear();
        filters.reserve(117); // 39 * (x,y,z) == 39 * 3 == 117
        for (int i = 0; i < 117; i++) {
            filters.emplace_back(f_win_size, f_v_scale, f_fps);
        }
        initialized = true;
        prediction = false;
    }

    PosePipelineOutput PosePipeline::pass(const cv::Mat &frame) {
        cv::Mat segmentation;
        return pass(frame, segmentation);
    }

    PosePipelineOutput PosePipeline::pass(const cv::Mat &frame, cv::Mat &segmented) {
        return inference(frame, segmented, nullptr);
    }

    PosePipelineOutput PosePipeline::pass(const cv::Mat &frame, cv::Mat &segmented, cv::Mat &debug) {
        return inference(frame, segmented, &debug);
    }

    PosePipelineOutput PosePipeline::inference(const cv::Mat &frame, cv::Mat &segmented, cv::Mat *debug) {
        if (!initialized) {
            init();
        }

        PosePipelineOutput output;
        cv::Mat source;

        if (prediction) {
            // crop using roi
            source = frame(cv::Rect(roi.x, roi.y, roi.w, roi.h));
        }

            // No prediction or to close to the border
        else {
            // using pose detector
            auto detections = detector.inference(frame);

            if (detections.empty() || detections[0].score < threshold_detector) {
                output.present = false;
                output.score = 0;

                if (debug)
                    frame.copyTo(*debug);

                return output;
            }

            _detector_score = detections[0].score;

            auto &detected = detections[0];

            auto &body = detected.body;
            body.x *= (float) frame.cols;
            body.y *= (float) frame.rows;
            body.w *= (float) frame.cols;
            body.h *= (float) frame.rows;

            body.x += roi_padding_x - (roi_margin / 2.f);
            body.y += roi_padding_y - (roi_margin / 2.f);
            body.w += (roi_margin / 2.f);
            body.h += (roi_margin / 2.f);

            body.c.x *= (float) frame.cols;
            body.c.y *= (float) frame.rows;
            body.e.x *= (float) frame.cols;
            body.e.y *= (float) frame.rows;

            body.c.x += roi_padding_x - (roi_margin / 2.f);
            body.c.y += roi_padding_y - (roi_margin / 2.f);
            body.e.x += roi_padding_x - (roi_margin / 2.f);
            body.e.y += roi_padding_y - (roi_margin / 2.f);

            auto &face = detected.face;
            face.x *= (float) frame.cols;
            face.y *= (float) frame.rows;
            face.w *= (float) frame.cols;
            face.h *= (float) frame.rows;

            roi = eox::dnn::clamp_roi(body, frame.cols, frame.rows);
            source = frame(cv::Rect(roi.x, roi.y, roi.w, roi.h));

            for (auto &filter: filters) {
                filter.reset();
            }
        }

        auto result = pose.inference(source);
        const auto now = timestamp();

        if (result.score > threshold_pose) {
            eox::dnn::Landmark landmarks[39];
            for (int i = 0; i < 39; i++) {
                landmarks[i] = {
                        // turning x,y into common (global) coordinates
                        .x = (result.landmarks_norm[i].x * roi.w) + roi.x,
                        .y = (result.landmarks_norm[i].y * roi.h) + roi.y,

                        // z is still normalized (in range of 0 and 1)
                        .z = result.landmarks_norm[i].z,

                        .v = result.landmarks_norm[i].v,
                        .p = result.landmarks_norm[i].p,
                };
            }

            // temporal filtering (low pass based on velocity (literally))
            for (int i = 0; i < 39; i++) {
                const auto idx = i * 3;

                auto fx = filters.at(idx + 0).filter(now, landmarks[i].x);
                auto fy = filters.at(idx + 1).filter(now, landmarks[i].y);
                auto fz = filters.at(idx + 2).filter(now, landmarks[i].z);

                landmarks[i].x = fx;
                landmarks[i].y = fy;
                landmarks[i].z = fz;
            }

            // perform segmentation
            if (segmentation()) {
                // if needed
                performSegmentation(result.segmentation, frame, segmented);
            } else {
                // or just use the very same frame
                segmented = frame;
            }

            // debug is a pointer actually, nullptr => false
            if (debug) {
                segmented.copyTo(*debug);
                drawJoints(landmarks, *debug);
                drawLandmarks(landmarks, result.landmarks_3d, *debug);
                drawRoi(*debug);
                printMetadata(result, *debug);
            }

            // saving old roi just in case
            const auto roi_old = roi;

            // predict new roi
            roi = roiPredictor
                    .setMargin(roi_margin)
                    .setFixX(roi_padding_x)
                    .setFixY(roi_padding_y)
                    .setScale(roi_scale)
                    .forward(eox::dnn::roiFromPoseLandmarks39(landmarks));
            _discarded_roi = false;

            // checking if we really need to use new roi
            if (prediction && roi_center_window > 0) {
                const float c_dist = std::sqrt(
                        std::pow(roi.c.x - roi_old.c.x, 2.f)
                        + std::pow(roi.c.y - roi_old.c.y, 2.f));
                const float radius = std::sqrt(
                        std::pow(roi_old.c.x - roi_old.e.x, 2.f)
                        + std::pow(roi_old.c.y - roi_old.e.y, 2.f));
                const float ratio = c_dist / radius;

                roi = (ratio >= roi_center_window)
                      ? roi
                      : roi_old;

                _discarded_roi = ratio < roi_center_window;
            }

            // clamping roi to prevent index out of range error
            const auto clamped_roi = eox::dnn::clamp_roi(roi, frame.cols, frame.rows);
            const auto ratio_roi_w = clamped_roi.w / roi.w;
            const auto ratio_roi_h = clamped_roi.h / roi.h;

            // checking if clamped roi big enough
            if (ratio_roi_w > roi_clamp_window && ratio_roi_h > roi_clamp_window) {
                // it is, we can use it
                roi = clamped_roi;
                prediction = true;
            } else {
                // it's not, gotta use detector
                prediction = false;
                _discarded_roi = true;
            }

            // preparing output
            {
                memcpy(output.segmentation, result.segmentation, 256 * 256 * sizeof(float));
                memcpy(output.ws_landmarks, result.landmarks_3d, 39 * sizeof(eox::dnn::Coord3d));
                memcpy(output.landmarks, landmarks, 39 * sizeof(eox::dnn::Landmark));

                output.score = result.score;
                output.present = true;
            }

        } else {

            // retry but without prediction
            if (prediction) {
                prediction = false;
                return inference(frame, segmented, debug);
            }

            // still nothing
            if (debug) {
                frame.copyTo(*debug);
                drawRoi(*debug);
            }
        }

        return output;
    }

    void PosePipeline::performSegmentation(float *segmentation_array, const cv::Mat &frame, cv::Mat &out) const {
        if (!segmentation()) {
            out = frame;
            return;
        }

        cv::Mat segmentation(256, 256, CV_32F, segmentation_array);
        cv::Mat segmentation_mask;

        cv::threshold(segmentation, segmentation_mask, 0.5, 1., cv::THRESH_BINARY);
        cv::resize(segmentation_mask, segmentation_mask, cv::Size(roi.w, roi.h));

        segmentation_mask.convertTo(segmentation_mask, CV_32FC1, 255.);
        segmentation_mask.convertTo(segmentation_mask, CV_8UC1);

        cv::Mat segmentation_frame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        segmentation_mask.copyTo(segmentation_frame(cv::Rect(roi.x, roi.y, roi.w, roi.h)));

        cv::bitwise_and(frame, frame, out, segmentation_frame);
    }

} // eox