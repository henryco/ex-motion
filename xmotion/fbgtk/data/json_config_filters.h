//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_FILTERS_H
#define XMOTION_JSON_CONFIG_FILTERS_H

#include "json_config_common.h"

namespace xm::data {

    typedef std::string FilterType;
#define XM_FILTER_TYPE_BLUR "blur"
#define XM_FILTER_TYPE_DIFF "difference"
#define XM_FILTER_TYPE_CHROMA "chromakey"

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

    typedef struct {
        // TODO
        bool _present;
    } Difference;

    typedef struct {
        int power;
        bool _present;
    } Blur;

    // Yes, I'm not going to practice polymorphic type gymnastics here
    typedef struct {
        xm::data::FilterType type;
        Blur blur;
        Chroma chroma;
        Difference difference;
    } Filter;
}

#endif //XMOTION_JSON_CONFIG_FILTERS_H
