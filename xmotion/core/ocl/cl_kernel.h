//
// Created by henryco on 6/6/24.
//

#ifndef XMOTION_CL_KERNEL_H
#define XMOTION_CL_KERNEL_H

#include <CL/cl.h>
#include <string>

namespace xm::ocl {

    /**
     * Computes global size for given problem dimension.
     *
     * \code
size_t l_size[2] = {32, 32};
size_t g_size[2] = {
     optimal_global_size(img_w, l_size[0]),
     optimal_global_size(img_h, l_size[1])
};
kernel.run(2, g_size, l_size, true)
     * \endcode
     *
     * @param dim global dimension size, ie. image dimension - width or height
     * @param local_size preferred local size for given GPU (often 32 or 64)
     * @return optimal global size
     */
    size_t optimal_global_size(int dim, size_t local_size);

    /**
     * Computes optimal local size (often multiplier like 32 or 64)
     */
    size_t optimal_local_size(cl_device_id device, cl_kernel kernel);

    /**
     * Creates new opencl context for given device
     */
    cl_context create_context(cl_device_id device);

    /**
     * Finds first available GPU device id
     */
    cl_device_id find_gpu_device();

    cl_program build_program(cl_context context, cl_device_id device, const std::string &kernel_source, const std::string &name);

    cl_program build_program(cl_context context, cl_device_id device, const char *source, size_t source_size, const std::string &name);

    cl_kernel build_kernel(cl_program program, const std::string &name);

    cl_command_queue create_queue_device(cl_context context, cl_device_id device, bool order, bool profile);

    cl_ulong measure_exec_time(cl_event event);

    /**
     * Sets kernel argument and returns incremented argument index.
     *
     * @example \code
cl_mem buffer = (cl_mem)umat_src.handle(cv::ACCESS_RW);
int width = 1920;
int height = 1080;

int idx = 0;
idx = set_kernel_arg(kernel, idx, sizeof(cl_mem), &buffer);
idx = set_kernel_arg(kernel, idx, sizeof(int), &width);
idx = set_kernel_arg(kernel, idx, sizeof(int), &height);
     * \endcode
     *
     * @param kernel opencl kernel
     * @param arg_index index of argument in kernel
     * @param arg_size size of the data
     * @param arg_value pointer to value/handle
     * @return Incremented arg_index
     */
    cl_uint set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, void *arg_value);

    cl_uint set_kernel_arg_ptr(cl_kernel kernel, cl_uint arg_index, void *svm_ptr);

    /**
     * @return Execution time in nanoseconds if profile is true
     */
    cl_ulong enqueue_kernel_sync(
            cl_command_queue command_queue,
            cl_kernel kernel,
            cl_uint work_dim,
            const size_t *global_work_size,
            const size_t *local_work_size,
            bool profile = false);

    /**
     * @return Enqueued kernel event
     */
    cl_event enqueue_kernel_fast(
            cl_command_queue command_queue,
            cl_kernel kernel,
            cl_uint work_dim,
            const size_t *global_work_size,
            const size_t *local_work_size,
            bool profile = false);

    void release_event(cl_event event);

    void finish_queue(cl_command_queue queue);

    bool check_svm_cap(cl_device_id device);

    /**
     * Checks platform and device for extension support
     * @param ext_name CL extension name, ie: "cl_khr_gl_sharing"
     * @param dev_id device id, optional
     */
    bool check_extension(const std::string &ext_name, cl_device_id dev_id = nullptr);

    /**
     * @return reference count for given memory object
     */
    cl_uint get_ref_count(cl_mem obj);
}

#endif //XMOTION_CL_KERNEL_H
