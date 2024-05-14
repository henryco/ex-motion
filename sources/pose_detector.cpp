//
// Created by henryco on 1/12/24.
//

#include "../xmotion/dnn/net/pose_detector.h"

#include <filesystem>
#include <opencv2/imgproc.hpp>

namespace eox::dnn {

    namespace dtc {
        const float *detector_bboxes_1x2254x12(const tflite::Interpreter &interpreter, box::Model model) {
            return interpreter.output_tensor(box::mappings[model].box_loc)->data.f;
        }

        const float *detector_scores_1x2254x1(const tflite::Interpreter &interpreter, box::Model model) {
            return interpreter.output_tensor(box::mappings[model].score_loc)->data.f;
        }
    }

//    const std::vector<std::string> PoseDetector::outputs = {
//            "Identity",   // 441  | 0: [1, 2254, 12]           un-decoded face bboxes location and key-points
//            "Identity_1", // 1429 | 4: [1, 2254, 1]            scores of the detected bboxes
//    };

    void PoseDetector::initialize() {
        anchors_vec = eox::dnn::ssd::generate_anchors(eox::dnn::ssd::SSDAnchorOptions(
                5,
                0.15,
                0.75,
                get_in_w(),
                get_in_h(),
                0.5,
                0.5,
                {8, 16, 32, 32, 32},
                {1.0},
                false,
                1.0,
                true
        ));
    }

    std::vector<eox::dnn::DetectedPose> PoseDetector::inference(cv::InputArray &frame) {
        auto ref = frame.getMat();
        view_w = ref.cols;
        view_h = ref.rows;
        cv::Mat blob = eox::dnn::convert_to_squared_blob(ref, get_in_w(), get_in_h(), true);

        with_box = true;
        const auto result = inference(blob.ptr<float>(0));
        with_box = false;
        return result;
    }

    std::vector<DetectedPose> PoseDetector::inference(const float *frame) {
        if (!with_box) {
            view_w = get_in_w();
            view_h = get_in_h();
        }

        init();
        input(0, frame, get_in_w() * get_in_h() * 3 * 4);
        invoke();

        // detection output
        std::vector<eox::dnn::DetectedPose> output;

        const auto bboxes = dtc::detector_bboxes_1x2254x12(*interpreter, model_type);
        const auto scores = dtc::detector_scores_1x2254x1(*interpreter, model_type);

        std::vector<float> scores_vec;
        std::vector<std::array<float, 12>> bboxes_vec;

        scores_vec.reserve(get_n_scores());
        bboxes_vec.reserve(get_n_bboxes());

        for (int i = 0; i < get_n_scores(); i++) {
            scores_vec.push_back(scores[i]);
        }

        for (int i = 0; i < get_n_bboxes(); i++) {
            std::array<float, 12> det_bbox{};
            for (int k = 0; k < 12; k++) {
                det_bbox[k] = bboxes[i * 12 + k];
            }
            bboxes_vec.push_back(det_bbox);
        }

        auto boxes = eox::dnn::ssd::decode_bboxes(
                threshold,
                scores_vec,
                bboxes_vec,
                anchors_vec,
                (float) get_in_w(),
                true);

        // correcting letterbox paddings
        const auto p = eox::dnn::get_letterbox_paddings(view_w, view_h, get_in_w(), get_in_h());
        const auto n_w = (float) get_in_w() - (p.left + p.right);
        const auto n_h = (float) get_in_h() - (p.top + p.bottom);

        for (auto &box: boxes) {
            eox::dnn::DetectedPose pose;

            {
                // face
                box.box.x = ((box.box.x * (float) get_in_w()) - p.left) / n_w;
                box.box.y = ((box.box.y * (float) get_in_h()) - p.top) / n_h;
                box.box.w = ((box.box.w * (float) get_in_w())) / n_w;
                box.box.h = ((box.box.h * (float) get_in_h())) / n_h;

                pose.face.x = box.box.x;
                pose.face.y = box.box.y;
                pose.face.w = box.box.w;
                pose.face.h = box.box.h;
            }

            {
                // body
                const float mid[2] = {box.key_points[0].x, box.key_points[0].y};
                const float end[2] = {box.key_points[1].x, box.key_points[1].y};

                auto body = roiPredictor
                        .setScale(roi_scale)
                        .setFixX(roi_padding_x)
                        .setFixY(roi_padding_y)
                        .setMargin(roi_margin)
                        .forward(eox::dnn::roiFromPoints(mid, end));

                body.x = ((body.x * (float) get_in_w()) - p.left) / n_w;
                body.y = ((body.y * (float) get_in_h()) - p.top) / n_h;
                body.w = ((body.w * (float) get_in_w())) / n_w;
                body.h = ((body.h * (float) get_in_h())) / n_h;
                pose.body = body;

                const float angle = M_PI * 0.5 - std::atan2(-(end[1] - mid[1]), end[0] - mid[0]);
                pose.rotation = (float) eox::dnn::normalize_radians(angle);
            }

            for (int i = 0; i < 4; i++) {
                // points
                auto &point = box.key_points[i];
                pose.points[i] = point;

                point.x = ((point.x * (float) get_in_w()) - p.left) / n_w;
                // problem was here, n_w instead of n_h
                point.y = ((point.y * (float) get_in_h()) - p.top) / n_h;
            }

            {
                // score
                pose.score = box.score;
            }

            output.push_back(pose);
        }

        return output;
    }

    std::string PoseDetector::get_model_file() {
        return "./../models/blazepose/body/detector/" + box::models[model_type];
    }

    float PoseDetector::getThreshold() const {
        return threshold;
    }

    void PoseDetector::setThreshold(float _threshold) {
        threshold = _threshold;
    }

    void PoseDetector::set_model_type(box::Model type) {
        model_type = type;
    }

    int PoseDetector::get_in_w() const {
        return box::mappings[model_type].i_w;
    }

    int PoseDetector::get_in_h() const {
        return box::mappings[model_type].i_h;
    }

    box::Model PoseDetector::get_model_type() const {
        return model_type;
    }

    int PoseDetector::get_n_bboxes() const {
        return box::mappings[model_type].bboxes;
    }

    int PoseDetector::get_n_scores() const {
        return box::mappings[model_type].scores;
    }

    void PoseDetector::setRoiScale(float scale) {
        roi_scale = scale;
    }

    float PoseDetector::getRoiScale() const {
        return roi_scale;
    }

    float PoseDetector::getRoiPaddingY() const {
        return roi_padding_y;
    }

    void PoseDetector::setRoiPaddingY(float roiPaddingY) {
        roi_padding_y = roiPaddingY;
    }

    float PoseDetector::getRoiPaddingX() const {
        return roi_padding_x;
    }

    void PoseDetector::setRoiPaddingX(float roiPaddingX) {
        roi_padding_x = roiPaddingX;
    }

    float PoseDetector::getRoiMargin() const {
        return roi_margin;
    }

    void PoseDetector::setRoiMargin(float roiMargin) {
        roi_margin = roiMargin;
    }

} // eox