//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_CALIBRATION_H
#define XMOTION_JSON_CONFIG_CALIBRATION_H

namespace xm::data {

    namespace board {
        enum Type {
            PLAIN = -1,
            CHESSBOARD,
            RADON
        };
    }

    typedef struct {
        board::Type type;
        int columns;
        int rows;
        float size;
    } Pattern;

    typedef struct {
        float x;
        float y;
        bool fix;
    } Intrinsic;

    typedef struct {
        Intrinsic f;
        Intrinsic c;
    } Intrinsics;

    typedef struct {
        std::vector<std::string> intrinsics;
        bool closed;
    } Chain;

    typedef struct {
        std::string name;
        Intrinsics intrinsics;
        Pattern pattern;
        Chain chain;
        int total;
        int delay;
    } Calibration;

    typedef struct {
        std::vector<std::string> files;
        bool closed;
        bool _present;
    } ChainCalibration;

    typedef struct {
        bool _present;
    } CrossCalibration;

}

#endif //XMOTION_JSON_CONFIG_CALIBRATION_H
