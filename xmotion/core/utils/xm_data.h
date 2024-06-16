//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_XM_DATA_H
#define XMOTION_XM_DATA_H

namespace xm::ds {
    typedef struct Color4u {
        unsigned char b = 0;
        unsigned char g = 0;
        unsigned char r = 0;
        unsigned char a = 255;

        unsigned char &h = b;
        unsigned char &s = g;
        unsigned char &l = r;
        unsigned char &v = r;

        unsigned char operator[](int idx) const;
        unsigned char operator[](int idx);

        static Color4u hsv(int h, int s, int v);
        static Color4u hsl(int h, int s, int l);
        static Color4u hls(int h, int l, int s);
        static Color4u bgr(int b, int g, int r);

        Color4u &operator=(const Color4u &other);
        Color4u &operator=(Color4u &&other) noexcept;

    } Color4u;
}

#endif //XMOTION_XM_DATA_H
