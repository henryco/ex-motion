//
// Created by henryco on 19/07/24.
//

#include "../../xmotion/core/pose/body/body_marker.h"

namespace xm::dnn::pose {

    BodyMarker::~BodyMarker() {
        if (inferencer)
            delete inferencer;
    }

    void BodyMarker::init(const ModelPose model) {
        if (inferencer)
            delete inferencer;
        inferencer = new platform::dnn::AgnosticBody(static_cast<platform::dnn::body::Model>((int) model));
    }

    void BodyMarker::inference(int n, const xm::ocl::iop::ClImagePromise *promises, eox::dnn::PoseOutput *poses, bool segmenation) {

        auto *mat_promises = new ocl::iop::CLPromise<cv::Mat>[n];
        for (int i = 0; i < n; i++)
            mat_promises[i] = xm::ocl::iop::to_cv_mat(promises[i]);

        xm::ocl::iop::CLPromise<cv::Mat>::finalizeAll(mat_promises, n);

        const auto in_w = inferencer->get_in_w();
        const auto in_h = inferencer->get_in_h();

        const auto m_dim = in_w * in_h;
        const auto m_size = m_dim * sizeof(float) * 3;

        auto batch_data = new float[n * m_dim * 3];

        for (int i = 0; i < n; i++) {
            const auto mat = eox::dnn::convert_to_squared_blob(
                mat_promises[i].get(),
                (int) in_w,
                (int) in_h,
                true);
            std::memcpy(batch_data + (i * n * 3), mat.data, m_size);
        }

        inferencer->inference(n, m_dim, batch_data);

        delete[] batch_data;
        delete[] mat_promises;

        const auto landmarks_3d = inferencer->get_landmarks_3d();
        const auto landmarks_wd = inferencer->get_landmarks_wd();
        const auto seg_masks    = inferencer->get_segmentations();
        const auto pose_flags   = inferencer->get_pose_flags();

        for (int i = 0; i < n; i++) {
            const auto img = promises[i].getImage2D();
            poses[i] = decode(
                landmarks_3d[i],
                landmarks_wd[i],
                seg_masks[i],
                pose_flags[i][0],
                (int) img.cols,
                (int) img.rows,
                segmenation);
        }
    }

    eox::dnn::PoseOutput BodyMarker::decode(
        const float *landmarks_3d,
        const float *landmarks_wd,
        const float *seg_mask,
        float        pose_flag,
        int          view_w,
        int          view_h,
        bool         segmentation
    ) {
        // TODO
        return {};
    }

}
