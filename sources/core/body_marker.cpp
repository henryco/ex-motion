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
        inferencer = new platform::dnn::AgnosticBody(static_cast<platform::dnn::body::Model>((int) model));
        // TODO
    }

    void BodyMarker::inference(int n, const xm::ocl::iop::ClImagePromise *frames, eox::dnn::PoseOutput *poses, bool segmenation) {
        // TODO
        // inferencer->inference(n, )
    }

}
