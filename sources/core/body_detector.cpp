//
// Created by henryco on 17/07/24.
//

#include "../../xmotion/core/pose/detector/body_detector.h"
#include "../../xmotion/core/dnn/net/ssd_anchors.h"

namespace xm::dnn::run {

    BodyDetector::~BodyDetector() {
        if (detector)
            delete detector;
    }

    void BodyDetector::init(const ModelDetector model) {
        if (detector)
            delete detector;
        detector = new platform::dnn::AgnosticDetector(static_cast<platform::dnn::detector::Model>((int) model));
        anchors_vec = eox::dnn::ssd::generate_anchors(eox::dnn::ssd::SSDAnchorOptions(
                5,
                0.15,
                0.75,
                (int) detector->get_in_w(),
                (int) detector->get_in_h(),
                0.5,
                0.5,
                {8, 16, 32, 32, 32},
                {1.0},
                false,
                1.0,
                true
        ));
    }

    // ReSharper disable once CppMemberFunctionMayBeConst
    void BodyDetector::detect(const int n, const xm::ocl::iop::ClImagePromise *promises, DetectedBody *detection) {

        auto *mat_promises = new ocl::iop::CLPromise<cv::Mat>[n];

        for (int i = 0; i < n; i++)
            mat_promises[i] = xm::ocl::iop::to_cv_mat(promises[i]);

        xm::ocl::iop::CLPromise<cv::Mat>::finalizeAll(mat_promises, n);

        const auto m_dim = detector->get_in_w() * detector->get_in_h() * 3;
        const auto m_size = m_dim * sizeof(float) * 3;

        auto batch_data = new float[n * m_dim * 3];

        for (int i = 0; i < n; i++) {
            // TODO: not very efficient, replace with ocl kernel on previous step
            const auto mat = eox::dnn::convert_to_squared_blob(
                    mat_promises[i].get(),
                    (int) detector->get_in_w(),
                    (int) detector->get_in_h(),
                    true);

            std::memcpy(batch_data + (i * n * 3), mat.data, m_size);
        }

        detector->inference(n, m_dim, batch_data);

        delete[] batch_data;
        delete[] mat_promises;

        const auto bboxes_a = detector->get_bboxes();
        const auto scores_a = detector->get_scores();

        for (int i = 0; i < n; i++) {
            const auto img = promises[i].getImage2D();
            detection[i] = decode(bboxes_a[i], scores_a[i], (int) img.cols, (int) img.rows);
        }
    }

    DetectedBody BodyDetector::decode(const float *bboxes, const float *scores, int view_w, int view_h) const {
        const auto in_w = detector->get_in_w();
        const auto in_h = detector->get_in_h();

        const auto n_bboxes = detector->get_n_bboxes();
        const auto n_scores = detector->get_n_scores();

        std::vector<float> scores_vec;
        std::vector<std::array<float, 12>> bboxes_vec;

        scores_vec.reserve(n_scores);
        bboxes_vec.reserve(n_bboxes);

        for (int i = 0; i < n_scores; i++) {
            scores_vec.push_back(scores[i]);
        }

        for (int i = 0; i < n_bboxes; i++) {
            std::array<float, 12> det_bbox{};
            for (int k = 0; k < 12; k++) {
                det_bbox[k] = bboxes[i * 12 + k];
            }
            bboxes_vec.push_back(det_bbox);
        }

        const auto boxes = eox::dnn::ssd::decode_bboxes(
                0,
                scores_vec,
                bboxes_vec,
                anchors_vec,
                (float) in_w,
                true);

        const auto p = eox::dnn::get_letterbox_paddings(view_w, view_h, (int) in_w, (int) in_h);
        const auto n_w = (float) in_w - (p.left + p.right);
        const auto n_h = (float) in_h - (p.top + p.bottom);

        if (boxes.empty()) {
            return { .score = 0.f };
        }

        const auto &box = boxes[0];
        const float mid[2] = {box.key_points[0].x, box.key_points[0].y};
        const float end[2] = {box.key_points[1].x, box.key_points[1].y};

        auto body = eox::dnn::to_roi(
                eox::dnn::from_array_xy(mid),
                eox::dnn::from_array_xy(end),
                margin,
                scale,
                padding_x,
                padding_y);

        body.x = ((body.x * (float) in_w) - p.left) / n_w;
        body.y = ((body.y * (float) in_h) - p.top) / n_h;
        body.w = ((body.w * (float) in_w)) / n_w;
        body.h = ((body.h * (float) in_h)) / n_h;

        body.x *= (float) view_w;
        body.y *= (float) view_h;
        body.w *= (float) view_w;
        body.h *= (float) view_h;

        body.x += padding_x - (margin / 2.f);
        body.y += padding_y - (margin / 2.f);
        body.w += (margin / 2.f);
        body.h += (margin / 2.f);

        body.c.x *= (float) view_w;
        body.c.y *= (float) view_h;
        body.e.x *= (float) view_w;
        body.e.y *= (float) view_h;

        body.c.x += padding_x - (margin / 2.f);
        body.c.y += padding_y - (margin / 2.f);
        body.e.x += padding_x - (margin / 2.f);
        body.e.y += padding_y - (margin / 2.f);

        return {
            .roi = body,
            .score = box.score
        };
    }

}