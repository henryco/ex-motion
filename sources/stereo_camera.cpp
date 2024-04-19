//
// Created by henryco on 4/19/24.
//

#include "../xmotion/camera/stereo_camera.h"

xm::StereoCamera::~StereoCamera() {
    release();
}

void xm::StereoCamera::release() {
    for (auto &capture: captures)
        capture.second.release();
    captures.clear();
}

void xm::StereoCamera::open() {

}