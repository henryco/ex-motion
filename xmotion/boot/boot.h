//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_BOOT_H
#define XMOTION_BOOT_H

/**
 * Interface class
 */
namespace xm {

    class Boot {
    public:
        virtual ~Boot()= default;

        virtual void open_project(const char *argv) = 0;

        virtual int boot(int &argc, char **&argv) = 0;
    };

} // xm

#endif //XMOTION_BOOT_H
