//
// Created by henryco on 19/07/24.
//

#ifndef AGNOSTIC_BODY_H
#define AGNOSTIC_BODY_H
#include <string>

#include "agnostic_dnn.h"

namespace platform::dnn {

    namespace body {
        enum Model {
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

        using Metadata = struct {
            int i_w;
            int i_h;
            int o_w;
            int o_h;
            int lm_3d;
            int seg;
            int flag;
        };

        extern const Metadata mappings[9];
        extern const std::string models[9];
    }

    class AgnosticBody {
    private:
        body::Model model = body::LITE_ORIGIN;
        AgnosticDnn *dnn_runner = nullptr;

        float **segmentations = nullptr;
        float **landmarks_3d = nullptr;
        float **pose_flags = nullptr;

        size_t batch = 0;

    public:
        explicit AgnosticBody(body::Model model);

        ~AgnosticBody();

        body::Model get_model() const;

        size_t get_in_w() const;

        size_t get_in_h() const;

        size_t get_n_lm3d() const;

        size_t get_n_segmentation_w() const;

        size_t get_n_segmentation_h() const;

        /**
         * @param in_batch_ptr float32[batch_size, N, N, 3] tensor (rgb image) flattened to 1D array.
         * Basicaly it's a 3D array (array of 2D images) flattened into 1D array.
         * \code
         * [i1_R1, i1_G1, i1_B1, i1_R2, i1_G2, i1_B2, ... , i2R1, i2G1, i2B1, ...]
         * \endcode
         */
        void inference(size_t batch_size, const float *in_batch_ptr);

        const float *const *get_segmentations() const;

        const float *const *get_landmarks_3d() const;

        const float *const *get_pose_flags() const;

    };

}

#endif //AGNOSTIC_BODY_H
