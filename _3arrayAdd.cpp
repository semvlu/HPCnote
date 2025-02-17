#include <CL/cl2.hpp>
#include <iostream>
#include <vector>

#define SIZE 10

const char* kern = R"(
 __kernel void _3arrAdd(global const int* A, global const int* B, global const int* C, global int* D) {
    D[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)] + C[get_global_id(0)];
}
)";



int main() {
    // ------ Sys env ------
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    cl::Platform plat = all_platforms[0];
    // Platform
    std::cout << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;


    std::vector<cl::Device> all_devices;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device dev = all_devices[0];
    std::cout << "Using device: " << dev.getInfo<CL_DEVICE_NAME>() << "\n";

    // ------ Create context & sources ------
    cl::Context contxt({ dev });
    cl::Program::Sources src;

    // ------ Input ------
    // Host 
    int A_h[SIZE] = { 0,1,2,3,4,5,6,7,8,9 };
    int B_h[SIZE] = { 10,9,8,7,6,5,4,3,2,1 };
    int C_h[SIZE] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288 };
    // ------ Buffer setup ------

    // Buffer: mem allo. to the dev.
    cl::Buffer buf_A(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer buf_B(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer buf_C(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);

    cl::Buffer buf_D(contxt, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);
    // CL_MEM_READ(WRITE)_ONLY / CL_MEM_READ_WRITE


    // ------ Command (Task) Queue ------
    // Queue: push cmd onto Dev, ~= CUDA streams
    cl::CommandQueue qu(contxt, dev);

    qu.enqueueWriteBuffer(buf_A, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    qu.enqueueWriteBuffer(buf_B, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
    qu.enqueueWriteBuffer(buf_C, CL_TRUE, 0, sizeof(int) * SIZE, C_h);
    // ------ Run kernel in source ------

    // push kern to the source code
    src.push_back({ kern, strlen(kern) });
    // OpenCL prog: link src to context
    cl::Program prog(contxt, src);

    // check build
    if (prog.build({ dev }) != CL_SUCCESS) {
        std::cout << " Error building: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << std::endl;
        exit(1);
    }
    // get_global_id, get_global_size, get_work_dim, get+
    // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/get_work_dim.html
    // https://community.khronos.org/t/when-to-use-get-global-id-and-get-local-id-in-opencl/3999/4


    // ------ Create kernel for exec ------
    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> kern0(cl::Kernel(prog, "_3arrAdd"));
    // *N.B.* Kernel name must match the function name

    // https://github.khronos.org/OpenCL-CLHPP/structcl_1_1compatibility_1_1make__kernel.html
    cl::NDRange global(SIZE); // #threads on dev

    kern0(cl::EnqueueArgs(qu, global),
        buf_A, buf_B, buf_C, buf_D).wait();


    int D_h[SIZE];

    // Retrive data from dev: from buf_D -> D_h
    qu.enqueueReadBuffer(buf_D, CL_TRUE, 0, sizeof(int) * SIZE, D_h);

    std::cout << "---------- Host D ----------" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << D_h[i] << std::endl;
    }
    std::cout << "----------------------------" << std::endl;

    return 0;
}