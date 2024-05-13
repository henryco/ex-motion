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
        constexpr float MARGIN = 30;
        constexpr float FIX_X = 0;
        constexpr float FIX_Y = 10;

        if (!initialized) {
            init();
        }

        PosePipelineOutput output;
        cv::Mat source;

        if (prediction) {
            // crop using roi
            roi = eox::dnn::clamp_roi(roi, frame.cols, frame.rows);
            source = frame(cv::Rect(roi.x, roi.y, roi.w, roi.h));
        }

        if (!prediction) {
            // using pose detector
            auto detections = detector.inference(frame);

            if (detections.empty() || detections[0].score < threshold_detector) {
                output.present = false;
                output.score = 0;

                if (debug)
                    frame.copyTo(*debug);

                return output;
            }

            auto &detected = detections[0];

            auto &body = detected.body;
            body.x *= (float) frame.cols;
            body.y *= (float) frame.rows;
            body.w *= (float) frame.cols;
            body.h *= (float) frame.rows;

            body.x += FIX_X - (MARGIN / 2.f);
            body.y += FIX_Y - (MARGIN / 2.f);
            body.w += (MARGIN / 2.f);
            body.h += (MARGIN / 2.f);

            body.c.x *= (float) frame.cols;
            body.c.y *= (float) frame.rows;
            body.e.x *= (float) frame.cols;
            body.e.y *= (float) frame.rows;

            body.c.x += FIX_X - (MARGIN / 2.f);
            body.c.y += FIX_Y - (MARGIN / 2.f);
            body.e.x += FIX_X - (MARGIN / 2.f);
            body.e.y += FIX_Y - (MARGIN / 2.f);

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

            performSegmentation(result.segmentation, frame, segmented);

            if (debug) {
                segmented.copyTo(*debug);
                drawJoints(landmarks, *debug);
                drawLandmarks(landmarks, result.landmarks_3d,*debug);
                drawRoi(*debug);
            }

            // predict new roi
            roi = roiPredictor
                    .setMargin(MARGIN)
                    .setFixX(FIX_X)
                    .setFixY(FIX_Y)
                    .forward(eox::dnn::roiFromPoseLandmarks39(landmarks));
            prediction = true;
//            prediction = false;

            // output
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