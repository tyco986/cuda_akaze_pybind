/**
 * Standalone test for gHammingMatch - loads saved inputs and runs kernel only.
 * Usage: ./test_gHammingMatch match_input.bin [output.txt]
 * Run 32 times and diff outputs to verify reproducibility.
 */
#include "akaze_structures.h"
#include "cuda_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>

#define X2 16
#define FLEN 61
#define MAX_DIST 96

/* Must match akaze_structures.h AkazePoint - use akaze namespace */
using AkazePoint = akaze::AkazePoint;

__device__ int dHammingDistance2(unsigned char* f1, unsigned char* f2) {
    int dist = 0;
    unsigned long long* v1 = (unsigned long long*)f1;
    unsigned long long* v2 = (unsigned long long*)f2;
    for (int i = 0; i < 8; i++)
        dist += __popcll(v1[i] ^ v2[i]);
    return dist;
}

__global__ void gHammingMatch(AkazePoint* points1, AkazePoint* points2, int n1, int n2) {
    __shared__ unsigned char ofeat[FLEN];
    __shared__ int distance[X2];
    __shared__ int indice[X2];
    __shared__ int flags[X2];

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (bid >= n1)
        return;

    AkazePoint* p1 = &points1[bid];
    AkazePoint* p2 = &points2[tid];

    if (tid == 0) {
        for (int i = 0; i < FLEN; i++)
            ofeat[i] = p1->features[i];
    }
    __syncthreads();

    distance[tid] = dHammingDistance2(ofeat, p2->features);
    indice[tid] = tid;
    __syncthreads();

    int max_iter = (n2 > X2) ? ((n2 - 1 - X2) / X2 + 1) : 0;
    for (int k = 0; k < max_iter; k++) {
        int pi = (k + 1) * X2 + tid;
        if (pi < n2) {
            p2 = &points2[pi];
            int dist = dHammingDistance2(ofeat, p2->features);
            if (dist < distance[tid]) {
                distance[tid] = dist;
                indice[tid] = pi;
            }
        }
        __syncthreads();
    }

    for (int stride = X2 / 2; stride > 0; stride >>= 1) {
        int ntid = tid + stride;
        if (tid < stride && distance[ntid] < distance[tid]) {
            int temp = distance[tid];
            distance[tid] = distance[ntid];
            distance[ntid] = temp;
            temp = indice[tid];
            indice[tid] = indice[ntid];
            indice[ntid] = temp;
        }
        __syncthreads();
    }

    flags[tid] = distance[0] < distance[tid] ? 1 : 0;
    __syncthreads();

    if (tid < 8) {
        volatile int* vsmem = flags;
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        if (flags[0] == (X2 - 1) && distance[0] < MAX_DIST) {
            p2 = &points2[indice[0]];
            p1->match = indice[0];
            p1->distance = distance[0];
            p1->match_x = p2->x;
            p1->match_y = p2->y;
        } else {
            p1->match = -1;
            p1->distance = -1;
            p1->match_x = -1;
            p1->match_y = -1;
        }
    }
    __syncthreads();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " match_input.bin [output.txt]\n";
        return 1;
    }
    const char* inpath = argv[1];
    const char* outpath = (argc >= 3) ? argv[2] : "match_output.txt";

    std::ifstream f(inpath, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open " << inpath << std::endl;
        return 1;
    }
    uint32_t n1, n2;
    f.read(reinterpret_cast<char*>(&n1), 4);
    f.read(reinterpret_cast<char*>(&n2), 4);
    std::vector<AkazePoint> h_pts1(n1), h_pts2(n2);
    f.read(reinterpret_cast<char*>(h_pts1.data()), sizeof(AkazePoint) * n1);
    f.read(reinterpret_cast<char*>(h_pts2.data()), sizeof(AkazePoint) * n2);
    f.close();

    initDevice(0);

    AkazePoint *d_pts1 = nullptr, *d_pts2 = nullptr;
    CHECK(cudaMalloc(&d_pts1, sizeof(AkazePoint) * n1));
    CHECK(cudaMalloc(&d_pts2, sizeof(AkazePoint) * n2));
    CHECK(cudaMemcpy(d_pts1, h_pts1.data(), sizeof(AkazePoint) * n1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pts2, h_pts2.data(), sizeof(AkazePoint) * n2, cudaMemcpyHostToDevice));

    dim3 block(X2);
    dim3 grid(n1);
    gHammingMatch<<<grid, block>>>(d_pts1, d_pts2, (int)n1, (int)n2);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_pts1.data(), d_pts1, sizeof(AkazePoint) * n1, cudaMemcpyDeviceToHost));

    std::ofstream out(outpath);
    out << "n1=" << n1 << " n2=" << n2 << "\n";
    for (uint32_t i = 0; i < n1 && i < 100; i++) {
        const auto& p = h_pts1[i];
        out << "pt" << i << ": match=" << p.match << " dist=" << p.distance
            << " match_x=" << p.match_x << " match_y=" << p.match_y << "\n";
    }
    if (n1 > 100)
        out << "... (truncated, total " << n1 << " pts)\n";
    out.close();

    CHECK(cudaFree(d_pts1));
    CHECK(cudaFree(d_pts2));
    cudaDeviceReset();
    std::cout << "Wrote " << outpath << " (n1=" << n1 << " n2=" << n2 << ")\n";
    return 0;
}
