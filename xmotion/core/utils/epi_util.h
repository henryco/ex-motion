//
// Created by henryco on 5/23/24.
//

#ifndef XMOTION_EPI_UTIL_H
#define XMOTION_EPI_UTIL_H

#include <opencv2/core/mat.hpp>

namespace xm::util::epi {

    typedef struct EpiPair {
        /**
         * Rotation-translation matrix. \n
         * Maps points from camera1 to camera2
         * world coordinate system: \n
         * X2 = RT * X1
         *
         * \code
         * ┌ R R R tx ┐
         * │ R R R ty │
         * │ R R R tz │
         * └ 0 0 0 1  ┘
         * \endcode
         */
        cv::Mat RT;

        /**
         * Essential matrix 3x3 \n
         * Maps points on image of camera1
         * to lines on image of camera2: \n
         * l_2 = F * x1
         */
        cv::Mat E;

        /**
         * Fundamental matrix 3x3 \n
         * Maps points on image of camera1
         * to lines on image of camera2: \n
         * l_2 = F * x1
         */
        cv::Mat F;
    } EpiPair;

    typedef struct CalibPair {

        /**
         * Camera1 calibration matrix 3x3
         *
         * \code
         * ┌ax    xo┐
         * │   ay yo│
         * └       1┘
         * \endcode
         */
        cv::Mat K1;

        /**
         * Camera2 calibration matrix 3x3
         *
         * \code
         * ┌ax    xo┐
         * │   ay yo│
         * └       1┘
         * \endcode
         */
        cv::Mat K2;

        /**
         * Rotation-translation matrix. \n
         * Maps points from camera1 to camera2
         * world coordinate system: \n
         * X2 = RT * X1
         *
         * \code
         * ┌ R R R tx ┐
         * │ R R R ty │
         * │ R R R tz │
         * └ 0 0 0 1  ┘
         * \endcode
         */
        cv::Mat RT;

        /**
         * Essential matrix 3x3 \n
         * Maps points on image of camera1
         * to lines on image of camera2: \n
         * l_2 = F * x1
         */
        cv::Mat E;

        /**
         * Fundamental matrix 3x3 \n
         * Maps points on image of camera1
         * to lines on image of camera2: \n
         * l_2 = F * x1
         */
        cv::Mat F;
    } CalibPair;

    /**
     * @param K1 3x3 calibration matrix for camera1
     * @param K2 3x3 calibration matrix for camera2
     * @param RT 4x4 Rotation-Translation matrix from camera1 to camera2
     */
    CalibPair calibrated_pair(const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &RT);

    /**
     * @param chain calibration chain [(0,1), (1,2), ..., (N,0)]
     * @param closed whether chain contains last pair (N,0)
     * @return chain oriented to origin: [(0,0), (1,0), (2,0), ... (N,0)]
     */
    std::vector<CalibPair> chain_to_origin(const std::vector<CalibPair> &chain, bool closed = false);

    /**
     * NxN matrix of stereo pairs (K1, K2, F, E and RT) \n
     * Pairs (f,t): (from -> to)
     *
     * \code
     *    t0 t1 t2    tN
     * f0 ┌X  .  .     .┐
     * f1 │.  X  .     .│
     * f2 │.  .  X     .│
     *    │             │
     * fN └.  .  .     X┘
     * \endcode
     */
    class Matrix {

    private:
        class Row {
        private:
            friend class Matrix;

            xm::util::epi::EpiPair *data;
            int size;
            int row;

            Row(xm::util::epi::EpiPair *data, int size, int row);

        public:
            xm::util::epi::EpiPair &operator[](int col);

            const xm::util::epi::EpiPair &operator[](int col) const;
        };

    public:
        /**
         * @note This is continuous flat array of the size of: NxN
         */
        xm::util::epi::EpiPair *epipolar_matrix = nullptr;

        /**
         * Size of one of the dimensions [ N ], \n
         * So the whole matrix size is [ N x N ]
         */
        int epipolar_matrix_size = 0;

        Matrix() = default;

        Matrix(xm::util::epi::EpiPair *data, int size);

        Matrix(const Matrix &src);

        Matrix(Matrix &&ref) noexcept;

        ~Matrix();

        Matrix& operator=(const Matrix& src);

        Row operator[](int row);

        const Row operator[](int row) const;

        [[nodiscard]] bool empty() const;

        [[nodiscard]] int rows() const;

        [[nodiscard]] int cols() const;

        [[nodiscard]] int size() const;

        void copyFrom(const Matrix &src);

        void release();

        [[nodiscard]] std::string to_string() const;

        /**
         * @param chain calibration chain [(X,Y), (Y,Z), ..., (Z,W)]
         * @param closed whether chain contains last pair (N,0)
         * @param origin whether chain is form of [(0,0), (1,0), (2,0), ..., (N,0)] (true)
         * or [(0,1), (1,2), (2,3), ... (N,0)] (false)
         * @return NxN matrix of stereo pairs (F, E and RT)
         */
        static Matrix from_chain(const std::vector<CalibPair> &chain, bool closed = false, bool origin = false);
    };


}

#endif //XMOTION_EPI_UTIL_H
