//
// Created by henryco on 1/7/24.
//

#ifndef STEREOX_ROI_PREDICTOR_H
#define STEREOX_ROI_PREDICTOR_H

#include "dnn_common.h"

namespace eox::dnn {
    class RoiPredictor {

    public:
        virtual eox::dnn::RoI forward(void* data) = 0;
    };
}

#endif //STEREOX_ROI_PREDICTOR_H
