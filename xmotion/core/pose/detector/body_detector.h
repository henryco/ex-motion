//
// Created by henryco on 17/07/24.
//

#ifndef XMOTION_BODY_DETECTOR_H
#define XMOTION_BODY_DETECTOR_H

#include "../../dnn/net/dnn_common.h"
#include "../../ocl/ocl_interop.h"
#include "../../../../platforms/agnostic_detector.h"

namespace xm::dnn::run {

    enum ModelDetector {
        ORIGIN = 0,
        F_32 = 1,
        F_16 = 2
    };

    using DetectedBody = struct {

        /**
         * RoI for body
         */
        eox::dnn::RoI roi;

        /**
         * Probability [0,1]
         */
        float score;
    };


    class BodyDetector {

    private:
        platform::dnn::AgnosticDetector *detector = nullptr;
        std::vector<std::array<float, 4>> anchors_vec;

    public:
        /**
         * Margins added to ROI
         */
        float margin = 0.f;

        /**
         * Horizontal paddings added to ROI
         */
        float padding_x = 0.f;

        /**
         * Vertical paddings added to ROI
         */
        float padding_y = 0.f;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float scale = 1.2f;

    public:
        BodyDetector() = default;

        ~BodyDetector();

        void init(ModelDetector model);

        void detect(int n, const xm::ocl::iop::ClImagePromise *frames, DetectedBody *detection);

    protected:
        DetectedBody decode(const float *bboxes, const float *scores, int view_w, int view_h) const;
    };

}

#endif //XMOTION_BODY_DETECTOR_H
