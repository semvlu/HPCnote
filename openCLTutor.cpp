#include <CL/cl2.hpp>
#include <iostream>
#include <vector>

#define SIZE 10

/*
    OpenCL tutorial by Universite du Luxembourg (UL) HPC
    https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/
    https://github.com/ULHPC/tutorials/tree/devel/gpu/opencl/code

    OpenCL Error code reference: https://gist.github.com/bmount/4a7144ce801e5569a0b6
*/

// Arch Hierarchy: Host=Global Mem > Device > Compute Unit (Work-group) > Processing Element (Work-item) 

// Kernel: string for hi portability, appended to prog @runtime
// const char* / std::string
// R"()": syntax for raw string
// kern func: void. global: global mem

const char* kern = R"(
 __kernel void kern0(global const int* A, global const int* B, global int* C) {
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
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

    // ------ Buffer setup ------

    // Buffer: mem allo. to the dev.
    cl::Buffer buf_A(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer buf_B(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    
    cl::Buffer buf_C(contxt, CL_MEM_READ_WRITE, sizeof(int) * SIZE);
    // CL_MEM_READ(WRITE)_ONLY / CL_MEM_READ_WRITE


    // ------ Command (Task) Queue ------
    // Queue: push cmd onto Dev, ~= CUDA streams
    cl::CommandQueue qu(contxt, dev);

    qu.enqueueWriteBuffer(buf_A, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    qu.enqueueWriteBuffer(buf_B, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
	
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
    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> kern0(cl::Kernel(prog, "kern0")); 
    // *N.B.* Kernel name must match the function name
   
    // https://github.khronos.org/OpenCL-CLHPP/structcl_1_1compatibility_1_1make__kernel.html
    cl::NDRange global(SIZE); // #threads on dev

    kern0( cl::EnqueueArgs(qu, global),
        buf_A, buf_B, buf_C).wait();


    int C_h[SIZE];

    // Retrive data from dev: from buf_C -> C_h
    qu.enqueueReadBuffer(buf_C, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

    std::cout << "---------- Host C ----------" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << C_h[i] << std::endl;
    }
    std::cout << "----------------------------" << std::endl;

    return 0;
}

