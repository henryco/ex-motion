//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_FILTERS_H
#define XMOTION_JSON_CONFIG_FILTERS_H

#include "json_config_common.h"

namespace xm::data {

    typedef struct {
        std::string key; // hex color
        std::string replace; // hex color
        HSL range;
        int blur;
        int power;
        int fine;
        int refine;
        bool linear;
        bool _present;
    } Chroma;

}

#endif //XMOTION_JSON_CONFIG_FILTERS_H
