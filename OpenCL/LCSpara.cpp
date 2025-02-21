#include <CL/cl2.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

const char* kern = R"(
__kernel void lcs_kern(__global const char* a, __global const char* b, const int m, const int n, __global int* tab) {
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    int idx = i * (n + 1) + j;
    int left = i * (n + 1) + j - 1;
    int diag = (i - 1) * (n + 1) + j - 1;
    int top  = (i - 1) * (n + 1) + j;
    
    if (b[j - 1] == a[i - 1])
        tab[idx] = tab[diag] + 1;
    else
        tab[idx] = max(tab[left], tab[top]);    
}
)";





std::vector<std::vector<int>> lcsPadding(int m, int n, std::vector<std::vector<int>> tab) {
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++)
            tab[i][j] = -1;
    }

    for (int i = 0; i <= m; i++)
        tab[i][0] = 0;
    for (int i = 0; i <= n; i++)
        tab[0][i] = 0;
    return tab;
}

std::string lcsSeq(std::string a, std::string b, std::vector<std::vector<int>> tab) {
    // Table traversal from the very last elem
    std::string seq;
    int r_trav = a.size();
    int c_trav = b.size(); 
    while (r_trav != 0 && c_trav != 0) {
        if (tab[r_trav][c_trav] > tab[r_trav - 1][c_trav - 1]) {
            seq += a[r_trav - 1];
            r_trav -= 1;
            c_trav -= 1;
        }

        else if (tab[r_trav][c_trav] == tab[r_trav - 1][c_trav])
            r_trav -= 1;
        else if (tab[r_trav][c_trav] == tab[r_trav][c_trav -1])
            c_trav -= 1;
    }
    std::reverse(seq.begin(), seq.end());
    return seq;
}



int main() {

    // ------ Sys env ------
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    
    cl::Platform plat = all_platforms[0];

    std::vector<cl::Device> all_devices;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device dev = all_devices[0];



    // ------ Create context & sources ------

    // Context: inter-device mem. share
    cl::Context contxt({ dev });
    cl::Program::Sources src;

    // ------ Input ------
    // Host 
    std::string A;
    std::string B;

    std::cout << "Enter sequence A: ";
    std::cin >> A;
    std::cout << "Enter sequence B: ";
    std::cin >> B;
    int m = A.size();
    int n = B.size();

    const char* A_h = A.c_str();
    const char* B_h = B.c_str();

    std::vector<std::vector<int>> tabVec(m + 1, std::vector<int>(n + 1));
    tabVec = lcsPadding(m, n, tabVec);

    int* tab = (int*) malloc(sizeof(int) * ((m + 1) * (n + 1)));
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            tab[i * (n + 1) + j] = tabVec[i][j];
        }
    }

    // ------ Buffer setup ------

    // Buffer: mem allo. to the dev.
    // Image: 2D/3D buffer
    cl::Buffer buf_A(contxt, CL_MEM_READ_ONLY, sizeof(char) * m);
    cl::Buffer buf_B(contxt, CL_MEM_READ_ONLY, sizeof(char) * n);
    cl::Buffer buf_m(contxt, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer buf_n(contxt, CL_MEM_READ_ONLY, sizeof(int));

    cl::Buffer buf_tab(contxt, CL_MEM_READ_WRITE, sizeof(int) * ((m + 1) * (n + 1)) );
    // CL_MEM_READ(WRITE)_ONLY / CL_MEM_READ_WRITE


    // ------ Command (Task) Queue ------
    // Queue: push cmd onto Dev, ~= CUDA streams
    // queue for each dev
    cl::CommandQueue qu(contxt, dev);
    // Read/Write/Map/Copy
    // blocking = CL_TRUE for sync
    qu.enqueueWriteBuffer(buf_A, CL_TRUE, 0, sizeof(char) * m, A_h);
    qu.enqueueWriteBuffer(buf_B, CL_TRUE, 0, sizeof(char) * n, B_h);
    qu.enqueueWriteBuffer(buf_m, CL_TRUE, 0, sizeof(int), &m);
    qu.enqueueWriteBuffer(buf_n, CL_TRUE, 0, sizeof(int), &n);

    qu.enqueueWriteBuffer(buf_tab, CL_TRUE, 0, sizeof(int) * ((m + 1) * (n + 1)), tab);

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

    // ------ Create kernel for exec ------
    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> lcs_kern(cl::Kernel(prog, "lcs_kern"));
    // *N.B.* Kernel name must match the function name
    // https://github.khronos.org/OpenCL-CLHPP/structcl_1_1compatibility_1_1make__kernel.html

    cl::NDRange global(m, n);
    
    // Event
    //cl::Event erLcsKern;

    lcs_kern(cl::EnqueueArgs(qu, global),
        buf_A, buf_B, buf_m, buf_n, buf_tab).wait();
    // .wait() or .waitForEvents( {ekern0} )

    //cl_ulong start = eLcsKern.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //cl_ulong end = eLcsKern.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    // Read data from dev
    qu.finish();
    qu.enqueueReadBuffer(buf_tab, CL_TRUE, 0, sizeof(int) * ((m + 1) * (n + 1)), tab);

    std::string buildLog = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++)
            tabVec[i][j] = tab[i * (n + 1) + j];
    }

    for (int i = 0; i < tabVec.size(); i++) {
        for (int j = 0; j < tabVec[i].size(); j++)
            std::cout << tabVec[i][j] << std::setw(2);
        std::cout << "\n";
    }

    std::string LCS = lcsSeq(A, B, tabVec);
    std::cout << "The LCS: " << LCS << std::endl;

    return 0;
}

