/**
 * Reproducibility Test - Runs AKAZE detection and matching, covering all paths in akazed.cu.
 * Covers: detectAndCompute (float), fastDetectAndCompute (uchar), cuMatch, setHistogram.
 */
#include "akaze.h"
#include "akazed.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define NBINS 300  /* Must match akazed.cu */

void dumpAkazeData(std::ofstream& out, const char* label, akaze::AkazeData& data) {
    out << "[" << label << "] num_pts=" << data.num_pts << " max_pts=" << data.max_pts << "\n";
    if (data.h_data && data.num_pts > 0) {
        for (int i = 0; i < data.num_pts && i < 50; i++) {
            akaze::AkazePoint& p = data.h_data[i];
            out << "  pt" << i << ": x=" << p.x << " y=" << p.y << " octave=" << p.octave
                << " response=" << p.response << " size=" << p.size << " angle=" << p.angle;
#if (FEATURE_TYPE == 5)
            out << " match=" << p.match << " dist=" << p.distance;
#endif
            out << "\n";
        }
        if (data.num_pts > 50) out << "  ... (truncated, total " << data.num_pts << " pts)\n";
    }
}

/* Descriptor checksum for reproducibility diagnosis - if this varies, issue is in gDescribe/gBuildDescriptor */
static uint32_t descChecksum(const unsigned char* f, int len) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) h = 31u * h + f[i];
    return h;
}

/* Save gHammingMatch inputs to file for standalone testing. Returns true if dump mode. */
bool dumpMatchInputIfRequested(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--dump-match-input") {
        const char* binpath = (argc >= 3) ? argv[2] : "match_input.bin";
        std::cout << "Dump mode: saving gHammingMatch inputs to " << binpath << std::endl;

        cv::Mat limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
        cv::Mat rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
        if (limg.empty() || rimg.empty()) {
            std::cerr << "Failed to load images" << std::endl;
            return true;
        }
        initDevice(0);
        { int h_hist[NBINS]; memset(h_hist, 0, sizeof(h_hist)); setHistogram(h_hist); }

        int3 whp, whp2;
        whp.x = limg.cols; whp.y = limg.rows; whp.z = iAlignUp(whp.x, 128);
        whp2.x = rimg.cols; whp2.y = rimg.rows; whp2.z = iAlignUp(whp2.x, 128);

        int max_npts = 10000, noctaves = 4, max_scale = 4;
        float per = 0.7f, kcontrast = 0.03f, soffset = 1.6f;
        bool reordering = true;
        float derivative_factor = 1.5f, dthreshold = 0.001f;
        int diffusivity = 1, descriptor_pattern_size = 10;

        auto detector = std::make_unique<akaze::Akazer>();
        detector->init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
                      derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

        cv::Mat fimg1, fimg2;
        limg.convertTo(fimg1, CV_32FC1, 1.0 / 255.0);
        rimg.convertTo(fimg2, CV_32FC1, 1.0 / 255.0);

        float* img1 = nullptr, *img2 = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(float) * whp.x, whp.y));
        CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(float) * whp2.x, whp2.y));
        CHECK(cudaMemcpy2D(img1, sizeof(float) * whp.z, fimg1.data, sizeof(float) * whp.x,
                          sizeof(float) * whp.x, whp.y, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy2D(img2, sizeof(float) * whp2.z, fimg2.data, sizeof(float) * whp2.x,
                          sizeof(float) * whp2.x, whp2.y, cudaMemcpyHostToDevice));

        akaze::AkazeData data1, data2;
        akaze::initAkazeData(data1, max_npts, true, true);
        akaze::initAkazeData(data2, max_npts, true, true);
        detector->detectAndCompute(img1, data1, whp, true);
        detector->detectAndCompute(img2, data2, whp2, true);

        std::ofstream f(binpath, std::ios::binary);
        uint32_t n1 = (uint32_t)data1.num_pts, n2 = (uint32_t)data2.num_pts;
        f.write(reinterpret_cast<const char*>(&n1), 4);
        f.write(reinterpret_cast<const char*>(&n2), 4);
        f.write(reinterpret_cast<const char*>(data1.h_data), sizeof(akaze::AkazePoint) * n1);
        f.write(reinterpret_cast<const char*>(data2.h_data), sizeof(akaze::AkazePoint) * n2);
        f.close();

        CHECK(cudaFree(img1));
        CHECK(cudaFree(img2));
        akaze::freeAkazeData(data1);
        akaze::freeAkazeData(data2);
        detector.reset();
        CHECK(cudaDeviceReset());
        std::cout << "Saved n1=" << n1 << " n2=" << n2 << std::endl;
        return true;
    }
    return false;
}

/* Save gHammingMatch inputs (data1, data2) to binary file. Same format as --dump-match-input. */
static void saveMatchInputToFile(const char* binpath, akaze::AkazeData& data1, akaze::AkazeData& data2) {
    std::ofstream f(binpath, std::ios::binary);
    uint32_t n1 = (uint32_t)data1.num_pts, n2 = (uint32_t)data2.num_pts;
    f.write(reinterpret_cast<const char*>(&n1), 4);
    f.write(reinterpret_cast<const char*>(&n2), 4);
    f.write(reinterpret_cast<const char*>(data1.h_data), sizeof(akaze::AkazePoint) * n1);
    f.write(reinterpret_cast<const char*>(data2.h_data), sizeof(akaze::AkazePoint) * n2);
    f.close();
}

/* Run path 3 block (detectAndCompute + cuMatch float). Returns true if ran. */
static bool runPath3(std::ofstream& out, const char* match_input_path,
    int3 whp, int3 whp2, int max_npts, cv::Mat& limg, cv::Mat& rimg,
    int noctaves, int max_scale, float per, float kcontrast, float soffset,
    bool reordering, float derivative_factor, float dthreshold, int diffusivity, int descriptor_pattern_size,
    std::function<std::unique_ptr<akaze::Akazer>()> makeDetector)
{
    auto detector = makeDetector();
    cv::Mat fimg1, fimg2;
    limg.convertTo(fimg1, CV_32FC1, 1.0 / 255.0);
    rimg.convertTo(fimg2, CV_32FC1, 1.0 / 255.0);

    float* img1 = nullptr;
    float* img2 = nullptr;
    size_t tmp_pitch = 0;
    CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(float) * whp.x, whp.y));
    CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(float) * whp2.x, whp2.y));
    CHECK(cudaMemcpy2D(img1, sizeof(float) * whp.z, fimg1.data, sizeof(float) * whp.x,
                      sizeof(float) * whp.x, whp.y, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(img2, sizeof(float) * whp2.z, fimg2.data, sizeof(float) * whp2.x,
                      sizeof(float) * whp2.x, whp2.y, cudaMemcpyHostToDevice));

    akaze::AkazeData data1, data2;
    akaze::initAkazeData(data1, max_npts, true, true);
    akaze::initAkazeData(data2, max_npts, true, true);

    detector->detectAndCompute(img1, data1, whp, true);
    detector->detectAndCompute(img2, data2, whp2, true);
    out << "[detectAndCompute_float_desc_checksum] ";
    for (int i = 0; i < 5 && i < data1.num_pts; i++)
        out << "pt" << i << "_h=" << descChecksum(data1.h_data[i].features, FLEN) << " ";
    for (int i = 0; i < 5 && i < data2.num_pts; i++)
        out << "pt2_" << i << "_h=" << descChecksum(data2.h_data[i].features, FLEN) << " ";
    out << "\n";
    if (match_input_path) {
        saveMatchInputToFile(match_input_path, data1, data2);
    }
    akaze::cuMatch(data1, data2);

    dumpAkazeData(out, "detectAndCompute_float_match1", data1);
    dumpAkazeData(out, "detectAndCompute_float_match2", data2);
    std::cout << "detectAndCompute + cuMatch (float): pts1=" << data1.num_pts
              << " pts2=" << data2.num_pts << std::endl;

    CHECK(cudaFree(img1));
    CHECK(cudaFree(img2));
    akaze::freeAkazeData(data1);
    akaze::freeAkazeData(data2);
    return true;
}

int main(int argc, char** argv) {
    if (dumpMatchInputIfRequested(argc, argv))
        return 0;

    const char* outfile = (argc > 1) ? argv[1] : "run_dump.txt";
    const char* match_input_path = (argc > 2) ? argv[2] : nullptr;
    /* Mode: default=full, path_order_inv=path3 first, path3_only=only path3 */
    std::string mode = (argc > 3) ? argv[3] : "full";
    std::cout << "Reproducibility test (" << mode << "): output -> " << outfile << std::endl;

    cv::Mat limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
    cv::Mat rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
    if (limg.empty()) {
        std::cerr << "Failed to load data/left.pgm" << std::endl;
        return 1;
    }
    if (rimg.empty()) {
        std::cerr << "Failed to load data/right.pgm" << std::endl;
        return 1;
    }

    initDevice(0);

    /* Cover setHistogram - called nowhere else in codebase */
    {
        int h_hist[NBINS];
        memset(h_hist, 0, sizeof(h_hist));
        setHistogram(h_hist);
    }

    int max_npts = 10000;
    int noctaves = 4, max_scale = 4;
    float per = 0.7f, kcontrast = 0.03f, soffset = 1.6f;
    bool reordering = true;
    float derivative_factor = 1.5f, dthreshold = 0.001f;
    int diffusivity = 1, descriptor_pattern_size = 10;

    int3 whp;
    whp.x = limg.cols;
    whp.y = limg.rows;
    whp.z = iAlignUp(whp.x, 128);

    int3 whp2;
    whp2.x = rimg.cols;
    whp2.y = rimg.rows;
    whp2.z = iAlignUp(whp2.x, 128);

    std::ofstream out(outfile);

    auto makeDetector = [&]() {
        auto d = std::make_unique<akaze::Akazer>();
        d->init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
               derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);
        return d;
    };

    auto runPath1 = [&]() {
        auto detector = makeDetector();
        cv::Mat fimg;
        limg.convertTo(fimg, CV_32FC1, 1.0 / 255.0);
        float* img = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&img, &tmp_pitch, sizeof(float) * whp.x, whp.y));
        const size_t dpitch = sizeof(float) * whp.z;
        const size_t spitch = sizeof(float) * whp.x;
        CHECK(cudaMemcpy2D(img, dpitch, fimg.data, spitch, spitch, whp.y, cudaMemcpyHostToDevice));
        akaze::AkazeData akaze_data;
        akaze::initAkazeData(akaze_data, max_npts, true, true);
        detector->detectAndCompute(img, akaze_data, whp, true);
        dumpAkazeData(out, "detectAndCompute_float", akaze_data);
        std::cout << "detectAndCompute (float): num_pts=" << akaze_data.num_pts << std::endl;
        CHECK(cudaFree(img));
        akaze::freeAkazeData(akaze_data);
    };

    auto runPath2 = [&]() {
        auto detector = makeDetector();
        unsigned char* img = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&img, &tmp_pitch, sizeof(unsigned char) * whp.x, whp.y));
        const size_t dpitch = sizeof(unsigned char) * whp.z;
        const size_t spitch = sizeof(unsigned char) * whp.x;
        CHECK(cudaMemcpy2D(img, dpitch, limg.data, spitch, spitch, whp.y, cudaMemcpyHostToDevice));
        akaze::AkazeData akaze_data;
        akaze::initAkazeData(akaze_data, max_npts, true, true);
        detector->fastDetectAndCompute(img, akaze_data, whp, true);
        dumpAkazeData(out, "fastDetectAndCompute_uchar", akaze_data);
        std::cout << "fastDetectAndCompute (uchar): num_pts=" << akaze_data.num_pts << std::endl;
        CHECK(cudaFree(img));
        akaze::freeAkazeData(akaze_data);
    };

    auto runPath4 = [&]() {
        auto detector = makeDetector();
        unsigned char* img1 = nullptr;
        unsigned char* img2 = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(unsigned char) * whp.x, whp.y));
        CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(unsigned char) * whp2.x, whp2.y));
        CHECK(cudaMemcpy2D(img1, sizeof(unsigned char) * whp.z, limg.data, sizeof(unsigned char) * whp.x,
                          sizeof(unsigned char) * whp.x, whp.y, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy2D(img2, sizeof(unsigned char) * whp2.z, rimg.data, sizeof(unsigned char) * whp2.x,
                          sizeof(unsigned char) * whp2.x, whp2.y, cudaMemcpyHostToDevice));
        akaze::AkazeData data1, data2;
        akaze::initAkazeData(data1, max_npts, true, true);
        akaze::initAkazeData(data2, max_npts, true, true);
        detector->fastDetectAndCompute(img1, data1, whp, true);
        detector->fastDetectAndCompute(img2, data2, whp2, true);
        akaze::cuMatch(data1, data2);
        dumpAkazeData(out, "fastDetectAndCompute_uchar_match1", data1);
        dumpAkazeData(out, "fastDetectAndCompute_uchar_match2", data2);
        std::cout << "fastDetectAndCompute + cuMatch (uchar): pts1=" << data1.num_pts
                  << " pts2=" << data2.num_pts << std::endl;
        CHECK(cudaFree(img1));
        CHECK(cudaFree(img2));
        akaze::freeAkazeData(data1);
        akaze::freeAkazeData(data2);
    };

    if (mode == "path3_only") {
        runPath3(out, match_input_path, whp, whp2, max_npts, limg, rimg,
            noctaves, max_scale, per, kcontrast, soffset, reordering,
            derivative_factor, dthreshold, diffusivity, descriptor_pattern_size, makeDetector);
    } else if (mode == "path_order_inv") {
        /* Path 3 first (no path 1/2 pollution), then path 1, 2, 4 */
        runPath3(out, match_input_path, whp, whp2, max_npts, limg, rimg,
            noctaves, max_scale, per, kcontrast, soffset, reordering,
            derivative_factor, dthreshold, diffusivity, descriptor_pattern_size, makeDetector);
        runPath1();
        runPath2();
        runPath4();
    } else {
        /* Default: path 1, 2, 3, 4 */
        runPath1();
        runPath2();
        runPath3(out, match_input_path, whp, whp2, max_npts, limg, rimg,
            noctaves, max_scale, per, kcontrast, soffset, reordering,
            derivative_factor, dthreshold, diffusivity, descriptor_pattern_size, makeDetector);
        runPath4();
    }

    out.close();

    CHECK(cudaDeviceReset());
    std::cout << "Done." << std::endl;
    return 0;
}
