//
// Created by henryco on 19/07/24.
//

#ifndef BODY_INFERENCE_H
#define BODY_INFERENCE_H
#include "../../../../platforms/agnostic_body.h"
#include "../../dnn/net/dnn_common.h"
#include "../../ocl/ocl_filters.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::dnn::run {

    enum ModelPose {
        HEAVY_ORIGIN = 0,
        FULL_ORIGIN = 1,
        LITE_ORIGIN = 2,

        HEAVY_F32 = 3,
        FULL_F32 = 4,
        LITE_F32 = 5,

        HEAVY_F16 = 6,
        FULL_F16 = 7,
        LITE_F16 = 8
    };

    class BodyMarker {
        static inline const auto log =
                spdlog::stdout_color_mt("body_marker");
    private:
        platform::dnn::AgnosticBody *inferencer = nullptr;

        ocl::iop::CLPromise<cv::Mat> *mat_promises = nullptr;
        float *batch_data = nullptr;
        int batch_size = 0;

    public:
        BodyMarker() = default;

        ~BodyMarker();

        void init(ModelPose model);

        void inference(int n, const xm::ocl::iop::ClImagePromise *frames, eox::dnn::PoseOutput *poses, bool segmenation = false);

    protected:
        eox::dnn::PoseOutput decode(
            const float *landmarks_3d,
            const float *seg_mask,
            float        pose_flag,
            int          view_w,
            int          view_h,
            bool         segmentation
        );

    };

}

#endif //BODY_INFERENCE_H
