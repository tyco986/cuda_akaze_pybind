/**
 * Reproducibility Test - Runs AKAZE detection twice and outputs structured data for comparison.
 * Used to identify which functions produce inconsistent output between runs.
 */
#include "akaze.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void runDetectionOnce(akaze::AkazeData& akaze_data, float* img, int3 whp, akaze::Akazer* detector) {
    detector->detectAndCompute(img, akaze_data, whp, true);
}

void dumpAkazeData(const char* filename, akaze::AkazeData& data) {
    std::ofstream out(filename);
    out << "num_pts=" << data.num_pts << "\n";
    out << "max_pts=" << data.max_pts << "\n";
    if (data.h_data && data.num_pts > 0) {
        for (int i = 0; i < data.num_pts && i < 100; i++) {
            akaze::AkazePoint& p = data.h_data[i];
            out << "pt" << i << ": x=" << p.x << " y=" << p.y << " octave=" << p.octave
                << " response=" << p.response << " size=" << p.size << " angle=" << p.angle;
#if (FEATURE_TYPE == 5)
            out << " feat=";
            for (int j = 0; j < 8 && j < FLEN; j++) out << (int)p.features[j] << ",";
            out << "...";
#endif
            out << "\n";
        }
        if (data.num_pts > 100) out << "... (truncated, total " << data.num_pts << " pts)\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    const char* outfile = (argc > 1) ? argv[1] : "run_dump.txt";
    std::cout << "Reproducibility test: output -> " << outfile << std::endl;

    cv::Mat limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
    if (limg.empty()) {
        std::cerr << "Failed to load data/left.pgm" << std::endl;
        return 1;
    }
    limg.convertTo(limg, CV_32FC1, 1.0 / 255.0);

    int3 whp;
    whp.x = limg.cols;
    whp.y = limg.rows;
    whp.z = iAlignUp(whp.x, 128);

    initDevice(0);

    float* img = nullptr;
    size_t tmp_pitch = 0;
    CHECK(cudaMallocPitch((void**)&img, &tmp_pitch, sizeof(float) * whp.x, whp.y));
    const size_t dpitch = sizeof(float) * whp.z;
    const size_t spitch = sizeof(float) * whp.x;
    CHECK(cudaMemcpy2D(img, dpitch, limg.data, spitch, spitch, whp.y, cudaMemcpyHostToDevice));

    int max_npts = 10000;
    int noctaves = 4, max_scale = 4;
    float per = 0.7f, kcontrast = 0.03f, soffset = 1.6f;
    bool reordering = true;
    float derivative_factor = 1.5f, dthreshold = 0.001f;
    int diffusivity = 1, descriptor_pattern_size = 10;

    akaze::AkazeData akaze_data;
    akaze::initAkazeData(akaze_data, max_npts, true, true);

    std::unique_ptr<akaze::Akazer> detector(new akaze::Akazer);
    detector->init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
                   derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

    runDetectionOnce(akaze_data, img, whp, detector.get());
    dumpAkazeData(outfile, akaze_data);
    std::cout << "num_pts=" << akaze_data.num_pts << std::endl;

    CHECK(cudaFree(img));
    akaze::freeAkazeData(akaze_data);
    CHECK(cudaDeviceReset());

    return 0;
}
