//
// Created by henryco on 19/07/24.
//

#include "../../xmotion/core/pose/body/body_marker.h"

namespace xm::dnn::run {

    BodyMarker::~BodyMarker() {
        if (inferencer)
            delete inferencer;
    }

    void BodyMarker::init(const ModelPose model) {
        if (inferencer)
            delete inferencer;
        inferencer = new platform::dnn::AgnosticBody(static_cast<platform::dnn::body::Model>((int) model));
    }

    void BodyMarker::inference(const int n, const xm::ocl::iop::ClImagePromise *promises, eox::dnn::PoseOutput *poses, const bool segmenation) {

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
            // TODO: not very efficient, replace with ocl kernel on previous step
            const auto mat = eox::dnn::convert_to_squared_blob(
                mat_promises[i].get(),
                (int) in_w,
                (int) in_h,
                true);
            std::memcpy(batch_data + (i * n * 3), mat.data, m_size);
        }

        inferencer->inference(n, batch_data);

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
        const float  pose_flag,
        const int    view_w,
        const int    view_h,
        const bool   segmentation
    ) {
        const auto in_w = inferencer->get_in_w();
        const auto in_h = inferencer->get_in_h();

        eox::dnn::PoseOutput output;
        output.score = pose_flag;

        const auto p = eox::dnn::get_letterbox_paddings(view_w, view_h, (int) in_w, (int) in_h);
        const auto n_w = (float) in_w - (p.left + p.right);
        const auto n_h = (float) in_h - (p.top + p.bottom);

        for (int i = 0; i < 39; i++) {
            const int j = i * 3;
            const int k = i * 5;
            // normalized landmarks_3d
            output.landmarks_norm[i] = {
                .x = (landmarks_3d[k + 0] - p.left) / n_w,
                .y = (landmarks_3d[k + 1] - p.top) / n_h,
                .z = landmarks_3d[k + 2] / (float) std::max(in_w, in_h),
                .v = landmarks_3d[k + 3],
                .p = landmarks_3d[k + 4],
            };

            // world-space landmarks
            output.landmarks_3d[i] = {
                .x = landmarks_wd[j + 0],
                .y = landmarks_wd[j + 1],
                .z = landmarks_wd[j + 2],
            };
        }

        if (!segmentation)
            return output;

        // There is a very important assumption that segmenation mask is always smaller or at least is the same as the original image

        const auto seg_w = (int) inferencer->get_n_segmentation_w();
        const auto seg_h = (int) inferencer->get_n_segmentation_h();

        const auto seg_p_w = (int) (((float) in_w - (float) n_w) / (float) in_w * (float) seg_w) / 2; // padding x
        const auto seg_p_h = (int) (((float) in_h - (float) n_h) / (float) in_h * (float) seg_h) / 2; // padding y

        const auto seg_t_w = seg_w - 2 * seg_p_w; // true segmentation width
        const auto seg_t_h = seg_h - 2 * seg_p_h;// true segmentation height

        const auto scale_x = (float) in_w / (float) seg_t_w;
        const auto scale_y = (float) in_h / (float) seg_t_h;
        const auto ceil_x  = (int) std::ceil(scale_x);
        const auto ceil_y  = (int) std::ceil(scale_y);

        const int seg_dim = seg_w * seg_h;

        for (int i = 0; i < seg_dim; i++) {
            const float frac_g = (float) i / (float) seg_w;
            const int   g_y    = (int) frac_g;
            const int   g_x    = (int) ((frac_g - (float) g_y) * (float) seg_w);

            if (g_x < seg_p_w || g_x >= seg_w - seg_p_w || g_y < seg_p_h || g_y >= seg_h - seg_p_h)
                continue; // filtering paddings

            const int x  = g_x - seg_p_w;
            const int y  = g_y - seg_p_h;
            const int ox = (int) ((float) x * scale_x);
            const int oy = (int) ((float) y * scale_y);

            const float sigmoid = (float) eox::dnn::sigmoid(seg_mask[i]);

            for (int ky = 0; ky < ceil_y; ky++) {
                for (int kx = 0; kx < ceil_x; kx++) {
                    const int ix = std::clamp(ox + kx, 0, (int) in_w - 1);
                    const int iy = std::clamp(oy + ky, 0, (int) in_h - 1);
                    const int odx = (iy * (int) in_w) + ix;
                    output.segmentation[odx] = sigmoid;
                }
            }
        }

        return output;
    }

}
