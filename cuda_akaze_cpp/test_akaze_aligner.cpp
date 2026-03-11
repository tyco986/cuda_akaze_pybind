/**
 * AkazeAligner benchmark: C++ equivalent of cuda_akaze/test_akaze_aligner.py
 * CLI: --image, --template, --batch, --device, --no-nndr, --mode
 * Output: GPU/CPU/memory peak, time, warp matrices
 */

#include "akaze_aligner.h"
#include "cuda_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __linux__
#include <unistd.h>
#include <sys/resource.h>
#endif

#if defined(HAVE_NVML) && HAVE_NVML
#include <nvml.h>
#define NVML_AVAILABLE 1
#else
#define NVML_AVAILABLE 0
#endif

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --image PATH      Input image (default: example_data/image.png)\n"
              << "  --template PATH   Template image (default: example_data/template.png)\n"
              << "  --batch N         Batch size (default: 8)\n"
              << "  --device N|cpu    CUDA device ID or 'cpu' for OpenCV AKAZE (default: 0)\n"
              << "  --no-nndr         Disable NNDR filter\n"
              << "  --mode fast|accurate  AKAZE mode (default: fast)\n";
}

static bool parseArgs(int argc, char** argv,
                     std::string& image_path, std::string& template_path,
                     int& batch, int& device, bool& use_nndr, std::string& mode) {
    image_path = "example_data/image.png";
    template_path = "example_data/template.png";
    batch = 8;
    device = 0;
    use_nndr = true;
    mode = "fast";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc)
            image_path = argv[++i];
        else if (arg == "--template" && i + 1 < argc)
            template_path = argv[++i];
        else if (arg == "--batch" && i + 1 < argc)
            batch = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) {
            std::string dev_str = argv[++i];
            if (dev_str == "cpu" || dev_str == "CPU")
                device = -1;
            else
                device = std::atoi(dev_str.c_str());
        }
        else if (arg == "--no-nndr")
            use_nndr = false;
        else if (arg == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

struct MonitorResult {
    std::atomic<uint64_t> gpu_peak{0};
    std::atomic<double> cpu_peak{0.0};
    std::atomic<uint64_t> mem_peak{0};
    std::atomic<bool> stop{false};
};

#ifdef __linux__
/* Read system-wide CPU usage from /proc/stat (aggregate of all cores). */
static void readSystemCpu(unsigned long long& total, unsigned long long& idle) {
    std::ifstream f("/proc/stat");
    std::string line;
    if (std::getline(f, line) && line.compare(0, 3, "cpu") == 0) {
        unsigned long long user, nice, system, iowait, irq, softirq, steal, guest, guest_nice;
        std::istringstream iss(line.substr(3));
        iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
        total = user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice;
    } else {
        total = idle = 0;
    }
}

static uint64_t readVmRss() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            uint64_t kb = 0;
            sscanf(line.c_str() + 6, "%lu", &kb);
            return kb * 1024;
        }
    }
    return 0;
}
#endif

static void monitorLoop(MonitorResult& res, int device_id) {
    const auto interval = std::chrono::milliseconds(5);

#if NVML_AVAILABLE
    nvmlDevice_t nvml_dev = nullptr;
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDeviceGetHandleByIndex(device_id, &nvml_dev);
    }
#endif

#ifdef __linux__
    unsigned long long total_prev = 0, idle_prev = 0;
    bool first = true;
#endif

    while (!res.stop.load()) {
#ifdef __linux__
        unsigned long long total, idle;
        readSystemCpu(total, idle);
        if (!first && total > total_prev) {
            unsigned long long delta_total = total - total_prev;
            unsigned long long delta_idle = idle - idle_prev;
            double cpu = 100.0 * (delta_total - delta_idle) / delta_total;
            double cur = res.cpu_peak.load();
            while (cpu > cur && !res.cpu_peak.compare_exchange_weak(cur, cpu)) {}
        }
        total_prev = total;
        idle_prev = idle;
        first = false;

        uint64_t rss = readVmRss();
        uint64_t cur_mem = res.mem_peak.load();
        while (rss > cur_mem && !res.mem_peak.compare_exchange_weak(cur_mem, rss)) {}
#endif

#if NVML_AVAILABLE
        if (nvml_dev) {
            nvmlMemory_t mem;
            if (nvmlDeviceGetMemoryInfo(nvml_dev, &mem) == NVML_SUCCESS) {
                uint64_t cur_gpu = res.gpu_peak.load();
                while (mem.used > cur_gpu && !res.gpu_peak.compare_exchange_weak(cur_gpu, mem.used)) {}
            }
        }
#endif

        std::this_thread::sleep_for(interval);
    }

#if NVML_AVAILABLE
    nvmlShutdown();
#endif
}

int main(int argc, char** argv) {
    std::string image_path, template_path, mode;
    int batch, device;
    bool use_nndr;
    if (!parseArgs(argc, argv, image_path, template_path, batch, device, use_nndr, mode)) {
        return 0;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat tpl = cv::imread(template_path, cv::IMREAD_GRAYSCALE);
    if (img.empty() || tpl.empty()) {
        std::cerr << "Failed to load images: " << image_path << ", " << template_path << "\n";
        return -1;
    }

    if (device >= 0) {
        initDevice(device);
    }

    std::vector<cv::Mat> templates(batch), images(batch);
    for (int b = 0; b < batch; b++) {
        templates[b] = tpl.clone();
        images[b] = img.clone();
    }

    std::vector<cv::Mat> warp_matrix;
    std::vector<cv::Vec2f> motion;
    double elapsed_ms = 0;
    MonitorResult mon;

    {
        akaze::AkazeAlignerConfig cfg;
        cfg.nndr = use_nndr ? 0.8f : 1.0f;
        cfg.mode = mode;
        akaze::AkazeAligner aligner(device, cfg);

        for (int w = 0; w < 3; w++) {
            std::vector<cv::Mat> H_out;
            std::vector<cv::Vec2f> mot_out;
            aligner.findTransformBatch(templates, images, H_out, mot_out);
        }

        std::thread monitor_thread(monitorLoop, std::ref(mon), device >= 0 ? device : 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        auto t0 = std::chrono::high_resolution_clock::now();
        aligner.findTransformBatch(templates, images, warp_matrix, motion);
        auto t1 = std::chrono::high_resolution_clock::now();

        mon.stop.store(true);
        monitor_thread.join();

        elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double gpu_mb = mon.gpu_peak.load() / (1024.0 * 1024.0);
    double mem_mb = mon.mem_peak.load() / (1024.0 * 1024.0);

    std::cout << "--- AkazeAligner benchmark ---\n";
    std::cout << "  Batch size:        " << batch << "\n";
    std::cout << "  Image shape:       (" << img.rows << ", " << img.cols << ")\n";
    std::cout << "  Template shape:    (" << tpl.rows << ", " << tpl.cols << ")\n";
    std::cout << "  Mode:              " << mode << "\n";
    std::cout << "  GPU peak (MB):     " << std::fixed << std::setprecision(2) << gpu_mb << "\n";
    std::cout << "  CPU peak (%):      " << std::fixed << std::setprecision(1) << mon.cpu_peak.load() << "\n";
    std::cout << "  Memory peak (MB):  " << std::fixed << std::setprecision(2) << mem_mb << "\n";
    std::cout << "  Time (ms):         " << std::fixed << std::setprecision(2) << elapsed_ms << "\n";
    std::cout << "  Time per sample:   " << std::fixed << std::setprecision(2) << (elapsed_ms / batch) << " ms\n";
    std::cout << "  Warp matrix (3x3 per sample):\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < batch && i < (int)warp_matrix.size(); i++) {
        std::cout << "    [" << i << "]:\n";
        const cv::Mat& H = warp_matrix[i];
        for (int r = 0; r < 3; r++) {
            std::cout << (r == 0 ? "[[ " : " [ ");
            for (int c = 0; c < 3; c++)
                std::cout << std::setw(8) << H.at<float>(r, c) << (c < 2 ? " " : "");
            std::cout << (r == 2 ? " ]]\n" : " ]\n");
        }
    }

    if (device >= 0)
        CHECK(cudaDeviceReset());
    return 0;
}
