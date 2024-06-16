//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/utils/xm_data.h"

#include <stdexcept>
#include <string>

namespace xm::ds {

    unsigned char Color4u::operator[](int idx) {
        if (idx == 0) return b;
        if (idx == 1) return g;
        if (idx == 2) return r;
        if (idx == 3) return a;
        throw std::out_of_range("color index out of range: " + std::to_string(idx));
    }

    unsigned char Color4u::operator[](int idx) const {
        if (idx == 0) return b;
        if (idx == 1) return g;
        if (idx == 2) return r;
        if (idx == 3) return a;
        throw std::out_of_range("color index out of range: " + std::to_string(idx));
    }

    Color4u Color4u::hsv(int h, int s, int v) {
        return bgr(h, s, v);
    }

    Color4u Color4u::hsl(int h, int s, int l) {
        return bgr(h, s, l);
    }

    Color4u Color4u::hls(int h, int l, int s) {
        return bgr(h, s, l);
    }

    Color4u Color4u::bgr(int b, int g, int r, int a) {
        return {
                .b = (unsigned char) b,
                .g = (unsigned char) g,
                .r = (unsigned char) r,
                .a = (unsigned char) a
        };
    }

    Color4u &Color4u::operator=(const Color4u &other) {
        if (this == &other)
            return *this;
        b = other.b;
        g = other.g;
        r = other.r;
        a = other.a;
        return *this;
    }

    Color4u &Color4u::operator=(Color4u &&other) noexcept {
        if (this == &other)
            return *this;
        b = other.b;
        g = other.g;
        r = other.r;
        a = other.a;
        return *this;
    }

}