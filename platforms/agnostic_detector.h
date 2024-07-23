//
// Created by henryco on 18/07/24.
//

#ifndef XMOTION_AGNOSTIC_DETECTOR_H
#define XMOTION_AGNOSTIC_DETECTOR_H

#include <string>
#include "agnostic_dnn.h"

namespace platform::dnn {

    namespace detector {
        enum Model {
            ORIGIN = 0,
            F_32 = 1,
            F_16 = 2
        };

        using Metadata = struct {
            int i_w;
            int i_h;
            int bboxes;
            int scores;
            int box_loc;
            int score_loc;
        };

        extern const Metadata mappings[3];
        extern const std::string models[3];
    }

    class AgnosticDetector {
    private:
        detector::Model model = detector::ORIGIN;
        AgnosticDnn *dnn_runner = nullptr;
        float **bboxes = nullptr;
        float **scores = nullptr;
        size_t batch = 0;

    public:
        explicit AgnosticDetector(detector::Model model);

        ~AgnosticDetector();

        detector::Model get_model() const;

        size_t get_in_w() const;

        size_t get_in_h() const;

        int get_n_bboxes() const;

        int get_n_scores() const;

        /**
         * @param in_batch_ptr float32[batch_size, N, N, 3] tensor (rgb image) flattened to 1D array.
         * Basicaly it's a 3D array (array of 2D images) flattened into 1D array.
         * \code
         * [i1_R1, i1_G1, i1_B1, i1_R2, i1_G2, i1_B2, ... , i2R1, i2G1, i2B1, ...]
         * \endcode
         */
        void inference(size_t batch_size, const float *in_batch_ptr);

        const float *const *get_bboxes() const;

        const float *const *get_scores() const;
    };

}

#endif //XMOTION_AGNOSTIC_DETECTOR_H
