//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_CALIBRATION_H
#define XMOTION_JSON_CONFIG_CALIBRATION_H

#include "json_config_common.h"

namespace xm::data {

    namespace board {
        enum Type {
            PLAIN = -1,  // same as chessboard
            CHESSBOARD,  // chessboard calibration pattern
            RADON        // radon calibration pattern
        };
    }

    typedef struct {
        /**
         * Calibration pattern
         */
        board::Type type;

        /**
         * Number of columns in calibration pattern
         */
        int columns;

        /**
         * Number of rows in calibration pattern
         */
        int rows;

        /**
         * Size of the calibration pattern square
         * (units are arbitrary, could be cm/mm/inches/etc)
         */
        float size;
    } Pattern;

    typedef struct {
        float x;
        float y;

        /**
         * Fix camera device intrinsics
         * (Don't try to optimise it during calibration)
         */
        bool fix;
    } Intrinsic;

    typedef struct {
        /**
         * Camera focal length "f" values (in px).
         * [Optional]
         */
        Intrinsic f;

        /**
         * Camera center "c" position (in px).
         * [Optional]
         */
        Intrinsic c;
    } Intrinsics;

    typedef struct {
        /**
         * Array of file names with devices intrinsic parameters,
         * ie: [ "calib_1.json", "calib_2.json", "calib_3.json" ]
         */
        FileNames intrinsics;

        /**
         * Is calibration chain closed
         * (last elements of the chain points to first)
         */
        bool closed;
    } Chain;

    typedef struct {
        /**
         * Calibration session name
         * (among others used for output file name)
         */
        std::string name;

        /**
         * Calibration intrinsic properties
         */
        Intrinsics intrinsics;

        /**
         * Calibration pattern properties
         */
        Pattern pattern;

        /**
         * Calibration chain properties.
         * Used when config type is "chain_calibration"
         */
        Chain chain;

        /**
         * Total frames used in calibration process.
         * Higher -> slower, but more precise
         */
        int total;

        /**
         * Delay in [ms], between consecutive frame shots
         * in calibration process
         */
        int delay;
    } Calibration;

    typedef struct {
        /**
         * Array of file names with devices intrinsic properties
         */
        FileNames files;

        /**
         * Is calibration chain closed
         * (last elements of the chain points to first)
         */
        bool closed;

        bool _present;
    } ChainCalibration;

    typedef struct {
        bool _present;
    } CrossCalibration;

}

#endif //XMOTION_JSON_CONFIG_CALIBRATION_H
