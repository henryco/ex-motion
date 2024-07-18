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
         * @param in_batch_ptr 2D array
         */
        void inference(size_t batch_size, size_t input_size, const float * const*in_batch_ptr);

        /**
         * @param in_batch_ptr 2D array FLATTENED INTO 1D
         */
        void inference(size_t batch_size, size_t input_size, const float *in_batch_ptr);

        const float *const *get_bboxes() const;

        const float *const *get_scores() const;
    };

}

#endif //XMOTION_AGNOSTIC_DETECTOR_H
