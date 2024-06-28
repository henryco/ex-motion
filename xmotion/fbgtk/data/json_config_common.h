//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_COMMON_H
#define XMOTION_JSON_CONFIG_COMMON_H

namespace xm::data {

    typedef struct {
        int x;
        int y;
        int w;
        int h;
    } Region;

    typedef struct {
        bool x;
        bool y;
    } Flip;

    typedef struct {
        float h;
        float s;
        float l;
    } HSL;

}

#endif //XMOTION_JSON_CONFIG_COMMON_H
