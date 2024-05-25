#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-const-return-type"
#pragma ide diagnostic ignored "readability-make-member-function-const"
//
// Created by henryco on 5/23/24.
//

#include "../../xmotion/core/utils/epi_util.h"
#include "../../xmotion/core/utils/cv_utils.h"
#include <regex>

namespace xm::util::epi {

    CalibPair calibrated_pair(const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &RT) {
        return {
            .K1 = K1.clone(),
            .K2 = K2.clone(),
            .RT = RT.clone(),
            .E = cv::Mat::eye(3, 3, CV_64F),
            .F = cv::Mat::eye(3, 3, CV_64F)
        };
    }

    Matrix::Row::Row(EpiPair *_data, int _size, int _row) {
        data = _data;
        size = _size;
        row = _row;
    }

    EpiPair &Matrix::Row::operator[](int col) {
        assert(col >= 0 && col < size);
        return data[(row * size) + col];
    }

    const EpiPair &Matrix::Row::operator[](int col) const {
        assert(col >= 0 && col < size);
        return data[(row * size) + col];
    }

    Matrix::Row Matrix::operator[](int row) {
        assert(row >= 0 && row < epipolar_matrix_size);
        return {epipolar_matrix, epipolar_matrix_size, row};
    }

    const Matrix::Row Matrix::operator[](int row) const {
        assert(row >= 0 && row < epipolar_matrix_size);
        return {epipolar_matrix, epipolar_matrix_size, row};
    }

    void Matrix::release() {
        if (epipolar_matrix == nullptr)
            return;
        // good old manual memory management
        delete[] epipolar_matrix;
        epipolar_matrix = nullptr;
    }

    void Matrix::copyFrom(const Matrix &src) {
        if (this == &src || src.epipolar_matrix == nullptr)
            return;

        epipolar_matrix_size = src.epipolar_matrix_size;
        epipolar_matrix = new EpiPair[epipolar_matrix_size * epipolar_matrix_size];
        for (int i = 0; i < epipolar_matrix_size; i++) {
            for (int j = 0; j < epipolar_matrix_size; j++) {
                const auto index = (i * epipolar_matrix_size) + j;
                epipolar_matrix[index] = {
                        .RT = src.epipolar_matrix[index].RT.clone(),
                        .E = src.epipolar_matrix[index].E.clone(),
                        .F = src.epipolar_matrix[index].F.clone()
                };
            }
        }
    }

    Matrix::Matrix(EpiPair *data, int size) {
        epipolar_matrix_size = size;
        epipolar_matrix = data;
    }

    Matrix::Matrix(const Matrix &src) {
        copyFrom(src);
    }

    Matrix::Matrix(Matrix &&ref) noexcept {
        epipolar_matrix_size = ref.epipolar_matrix_size;
        epipolar_matrix = ref.epipolar_matrix;
        ref.epipolar_matrix_size = 0;
        ref.epipolar_matrix = nullptr;
    }

    Matrix::~Matrix() {
        release();
    }

    Matrix &Matrix::operator=(const Matrix &src) {
        if (this == &src)
            return *this;
        release();
        copyFrom(src);
        return *this;
    }

    int Matrix::rows() const {
        return epipolar_matrix_size;
    }

    int Matrix::cols() const {
        return epipolar_matrix_size;
    }

    int Matrix::size() const {
        return epipolar_matrix_size * epipolar_matrix_size;
    }

    bool Matrix::empty() const {
        return epipolar_matrix_size == 0 || epipolar_matrix == nullptr;
    }

    std::vector<CalibPair> chain_to_origin(const std::vector<CalibPair> &chain, bool closed) {
        const int size = (int) chain.size() - (closed ? 1 : 0);

        // pairs (from, to): RT_from -> to
        // open:   [(0,1), (1,2), (2,3)]         | 3
        // closed: [(0,1), (1,2), (2,3), (3,0)]  | 4
        const auto &pairs = chain;

        // pairs (from, origin): RT_from -> origin
        // [(0,0), (1,0), (2,0), (3,0)]
        std::vector<CalibPair> relative;
        relative.reserve(size + 1);

        // real first pair, just identity matrix, (0,0): 0 -> 0
        relative.push_back({
            .K1 = pairs.front().K1.clone(),
            .K2 = pairs.front().K1.clone(),
            .RT = cv::Mat::eye(4, 4, CV_64F)
        });

        // prepare RT relative to first device
        // ->       [ |(0,1), (1,2), (2,3)|, <3,0>]  | pairs
        // <- [<0,0>, |(1,0), (2,0), (3,0)| ]        | relative
        for (int i = 0; i < size; i++) {

            if (i == 0) {
                // first pair, swap (0,1) -> (1,0)
                relative.push_back({
                    .K1 = pairs.front().K2.clone(),
                    .K2 = pairs.front().K1.clone(),
                    .RT = pairs.front().RT.clone().inv()
                });
                continue;
            }

            if (closed && i == size - 1) {
                // last element within closed chain (loop)
                // <3,0> -> (3,0)
                // [..., (3,0)]
                relative.push_back({
                    .K1 = pairs.back().K1.clone(),
                    .K2 = pairs.back().K2.clone(),
                    .RT = pairs.back().RT.clone()
                });
                break;
            }

            // within chain: RT_J0 = RT_10 * RT_21 * RT_32 * ... * RT_JI
            // 0: J -> I -> ... -> 3 -> 2 -> 1 -> 0

            // current RT matrix, swap (1,2) -> (2,1)
            const cv::Mat RT_i = pairs.at(i).RT.clone().inv();

            // previous RT matrix (1,0)
            const cv::Mat RT_p = relative.back().RT.clone();

            // (2,0): (2,1) -> (1,0)
            // RT_20 = RT_10 * RT_21
            const cv::Mat RT = RT_p * RT_i;

            //    pairs(1,2) -> 2
            const cv::Mat K1 = pairs.at(i).K2.clone();

            // relative(1,0) -> 0
            const cv::Mat K2 = relative.back().K2.clone();

            relative.push_back({
                .K1 = K1,
                .K2 = K2,
                .RT = RT
            });
        }

        return relative;
    }

    Matrix Matrix::from_chain(const std::vector<CalibPair> &chain, bool closed, bool origin) {

        // [(0,0), (1,0), (2,0), (3,0), ...]
        const auto &devices = !origin
                ? xm::util::epi::chain_to_origin(chain, closed)
                : chain;

        const int size = (int) devices.size();

        /*
         *    t0 t1 t2
         * f0 ┌X  .  .┐
         * f1 │.  X  .│
         * f2 └.  .  X┘
         */
        auto *epipolar_matrix = new EpiPair[size * size];

        for (int i = 0; i < size; i++) {
            // i: from device

            for (int j = 0; j < size; j++) {
                // j: to device

                // ie: (3,0): 3 -> 0
                const cv::Mat RT_i = devices.at(i).RT.clone();

                // ie: (1,0): 1 -> 0
                const cv::Mat RT_j = devices.at(j).RT.clone();

                // FROM -> origin -> TO
                // (3,1): (3,0) -> (0,1)
                // RT_31 = RT_01      * RT_30
                // RT_31 = inv(RT_10) * RT_30
                const cv::Mat RT = RT_j.inv() * RT_i;

                // Computing essential and fundamental matrix according to FIRST device within the chain
                // R and T decomposition of RT
                const auto R = RT(cv::Rect(0, 0, 3, 3)).clone();
                const auto T = RT.col(3).clone();

                // Elements of Translation vector
                const auto Tx = T.at<double>(0);
                const auto Ty = T.at<double>(1);
                const auto Tz = T.at<double>(2);

                // Skew-symmetric matrix of vector To
                const cv::Mat T_x = (cv::Mat_<double>(3, 3) << 0, -Tz, Ty, Tz, 0, -Tx, -Ty, Tx, 0);

                // Calibration matrices (3x3)
                const cv::Mat K_i = devices.at(i).K1.clone(); // FROM
                const cv::Mat K_j = devices.at(j).K1.clone(); // TO

                // Essential matrix
                const cv::Mat E = T_x * R;

                // Fundamental matrix: l_j = F_ij * x_i
                const cv::Mat F = (K_j.inv().t() * E * K_i.inv()).t();

                // Real matrix is flat continuous array
                epipolar_matrix[(i * size) + j] = {
                        .RT = RT,
                        .E = E,
                        .F = F
                };
            }
        }

        return {epipolar_matrix, size};
    }

    std::string Matrix::to_string() const {
        // [
        //   [i, j]: {Rt:[], E:[], F:[]}
        // ]
        std::string str;
        str += "\n[";
        for (int i = 0; i < epipolar_matrix_size; i++) {
            for (int j = 0; j < epipolar_matrix_size; j++) {
                const auto &data = this[0][i][j];
                str += "\n[" + std::to_string(i) + "," + std::to_string(j) + "]: ";
                str += "{Rt:" + std::regex_replace(xm::ocv::print_matrix(data.RT), std::regex("\n"), "") + ", ";
                str += "E:" + std::regex_replace(xm::ocv::print_matrix(data.E), std::regex("\n"), "") + ", ";
                str += "F:" + std::regex_replace(xm::ocv::print_matrix(data.F), std::regex("\n"), "") + "}";
            }
        }
        str += "\n]";
        return str;
    }

}

#pragma clang diagnostic pop