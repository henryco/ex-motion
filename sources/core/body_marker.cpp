//
// Created by henryco on 19/07/24.
//

#include "../../xmotion/core/pose/body/body_marker.h"
#include "../../xmotion/core/ocl/ocl_filters.h"

namespace xm::dnn::run {

    BodyMarker::~BodyMarker() {
        if (inferencer)
            delete inferencer;
        if (batch_data)
            delete[] batch_data;
        if (mat_promises)
            delete[] mat_promises;
    }

    void BodyMarker::init(const ModelPose model) {
        if (inferencer)
            delete inferencer;
        inferencer = new platform::dnn::AgnosticBody(static_cast<platform::dnn::body::Model>((int) model));
    }

    void BodyMarker::inference(
        const int                           n,
        const xm::ocl::iop::ClImagePromise *promises,
        eox::dnn::PoseOutput *              poses,
        const bool                          segmenation
    ) {
        const auto in_w = inferencer->get_in_w();
        const auto in_h = inferencer->get_in_h();

        const auto m_dim  = in_w * in_h;

        if (mat_promises == nullptr || batch_size < n) {
            if (mat_promises != nullptr)
                delete[] mat_promises;
            mat_promises = new ocl::iop::CLPromise<cv::Mat>[n];
        }

        if (batch_data == nullptr || batch_size < n) {
            if (batch_data != nullptr)
                delete[] batch_data;
            batch_data = new float[n * m_dim * 3];
        }

        batch_size = n;

        // const auto t0 = std::chrono::system_clock::now();

        for (int i = 0; i < n; i++) {
            const auto &p = eox::dnn::get_letterbox_paddings((int) promises[i].getImage2D().cols,
                                                            (int) promises[i].getImage2D().rows,
                                                            (int) in_w,
                                                            (int) in_h);
            const auto &promise = xm::ocl::letterbox_rgb_f32(promises[i],
                                                      p.width,
                                                      p.height,
                                                      (int) p.left,
                                                      (int) p.bottom,
                                                      true);
            mat_promises[i] = xm::ocl::iop::to_cv_mat(promise,
                                                      CV_32FC3);
        }

        xm::ocl::iop::CLPromise<cv::Mat>::finalizeAll(mat_promises, n);

        // const auto t1 = std::chrono::system_clock::now();
        // log->info("T: {}", duration_cast<std::chrono::nanoseconds>((t1 - t0)).count());

        for (int i = 0; i < n; i++) {
            const auto &mat = mat_promises[i].get();
            std::memcpy(batch_data + (i * m_dim * 3),
                        mat.data,
                        m_dim * 3 * sizeof(float));
        }

        inferencer->inference(n, batch_data);

        const auto landmarks_3d = inferencer->get_landmarks_3d();
        const auto seg_masks    = inferencer->get_segmentations();
        const auto pose_flags   = inferencer->get_pose_flags();

        for (int i = 0; i < n; i++) {
            const auto img = promises[i].getImage2D();
            poses[i]       = decode(
                                    landmarks_3d[i],
                                    seg_masks[i],
                                    pose_flags[i][0],
                                    (int) img.cols,
                                    (int) img.rows,
                                    segmenation);
        }
    }

    eox::dnn::PoseOutput BodyMarker::decode(
        const float *landmarks_3d,
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
            const int k = i * 5;
            // normalized landmarks_3d
            output.landmarks_norm[i] = {
                .x = (landmarks_3d[k + 0] - p.left) / n_w,
                .y = (landmarks_3d[k + 1] - p.top) / n_h,
                .z = landmarks_3d[k + 2] / (float) std::max(in_w, in_h),
                .v = landmarks_3d[k + 3],
                .p = landmarks_3d[k + 4],
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
