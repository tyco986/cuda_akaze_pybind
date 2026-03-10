#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "akaze.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

namespace py = pybind11;


namespace {
    constexpr int NOCTAVES = 4;
    constexpr int MAX_SCALE = 4;
    constexpr float PER = 0.7f;
    constexpr float KCONTRAST = 0.03f;
    constexpr float SOFFSET = 1.6f;
    constexpr bool REORDERING = true;
    constexpr float DERIVATIVE_FACTOR = 1.5f;
    constexpr float DTHRESHOLD = 0.001f;
    constexpr int DIFFUSIVITY = 1;  // PM_G2
    constexpr int DESCRIPTOR_PATTERN_SIZE = 10;

    bool parseUseFast(const std::string& mode) {
        if (mode == "fast") return true;
        if (mode == "accurate") return false;
        throw std::runtime_error("mode must be 'fast' or 'accurate', got: " + mode);
    }
}

struct AkazerPipeline {
    akaze::Akazer akazer;
    akaze::AkazeData data1, data2;
    uint8_t* d_img = nullptr;
    float* d_img_float = nullptr;
    size_t img_pitch = 0;
    size_t float_img_pitch = 0;
    int buf_width = 0, buf_height = 0;
    int max_pts;
    bool needs_init = true;
    int c_width = 0, c_height = 0;

    AkazerPipeline(int _max_pts = 10000) : max_pts(_max_pts) {
        std::memset(&data1, 0, sizeof(data1));
        std::memset(&data2, 0, sizeof(data2));
        akaze::initAkazeData(data1, max_pts, true, true);
        akaze::initAkazeData(data2, max_pts, true, true);
    }

    ~AkazerPipeline() {
        if (d_img) cudaFree(d_img);
        if (d_img_float) cudaFree(d_img_float);
        akaze::freeAkazeData(data1);
        akaze::freeAkazeData(data2);
    }

    AkazerPipeline(const AkazerPipeline&) = delete;
    AkazerPipeline& operator=(const AkazerPipeline&) = delete;

    void ensureImageBuffer(int w, int h) {
        if (w > buf_width || h > buf_height) {
            if (d_img) cudaFree(d_img);
            int alloc_w = std::max(w, buf_width);
            int alloc_h = std::max(h, buf_height);
            CHECK(cudaMallocPitch((void**)&d_img, &img_pitch, sizeof(uint8_t) * alloc_w, alloc_h));
            buf_width = alloc_w;
            buf_height = alloc_h;
        }
    }

    void ensureFloatImageBuffer(int w, int h) {
        if (w > buf_width || h > buf_height) {
            if (d_img_float) cudaFree(d_img_float);
            int alloc_w = std::max(w, buf_width);
            int alloc_h = std::max(h, buf_height);
            int pitch_elems = iAlignUp(alloc_w, 128);
            float_img_pitch = sizeof(float) * pitch_elems;
            CHECK(cudaMalloc((void**)&d_img_float, float_img_pitch * alloc_h));
            buf_width = alloc_w;
            buf_height = alloc_h;
        }
    }

    void initIfNeeded(int3 whp) {
        if (needs_init || c_width != whp.x || c_height != whp.y) {
            akazer.init(whp, NOCTAVES, MAX_SCALE, PER, KCONTRAST, SOFFSET,
                        REORDERING, DERIVATIVE_FACTOR, DTHRESHOLD, DIFFUSIVITY, DESCRIPTOR_PATTERN_SIZE);
            c_width = whp.x;
            c_height = whp.y;
            needs_init = false;
        }
    }

    void copyU8ToFloatAndUpload(const uint8_t* src, int w, int h, float* d_dst, int3 whp) {
        std::vector<float> tmp(static_cast<size_t>(w) * h);
        for (int i = 0; i < w * h; i++)
            tmp[i] = src[i] / 255.0f;
        size_t spitch = sizeof(float) * w;
        size_t dpitch = sizeof(float) * whp.z;
        CHECK(cudaMemcpy2D(d_dst, dpitch, tmp.data(), spitch, spitch, h, cudaMemcpyHostToDevice));
    }

    py::tuple detectAndMatch(
        py::array_t<uint8_t> img1, py::array_t<uint8_t> img2, float nndr, const std::string& mode)
    {
        bool use_fast = parseUseFast(mode);
        auto buf1 = img1.request(), buf2 = img2.request();
        if (buf1.ndim != 2 || buf2.ndim != 2)
            throw std::runtime_error("Images must be 2D (height, width)");

        int h1 = buf1.shape[0], w1 = buf1.shape[1];
        int h2 = buf2.shape[0], w2 = buf2.shape[1];

        int3 whp1{w1, h1, iAlignUp(w1, 128)};
        int3 whp2{w2, h2, iAlignUp(w2, 128)};

        initIfNeeded(whp1);

        if (use_fast) {
            ensureImageBuffer(std::max(w1, w2), std::max(h1, h2));

            size_t spitch1 = sizeof(uint8_t) * w1;
            CHECK(cudaMemcpy2D(d_img, img_pitch, buf1.ptr, spitch1, spitch1, h1, cudaMemcpyHostToDevice));
            data1.num_pts = 0;
            akazer.fastDetectAndCompute(d_img, data1, whp1, true);

            bool diff_size = (w2 != w1 || h2 != h1);
            if (diff_size) {
                initIfNeeded(whp2);
                needs_init = true;
            }

            size_t spitch2 = sizeof(uint8_t) * w2;
            CHECK(cudaMemcpy2D(d_img, img_pitch, buf2.ptr, spitch2, spitch2, h2, cudaMemcpyHostToDevice));
            data2.num_pts = 0;
            akazer.fastDetectAndCompute(d_img, data2, whp2, true);
        } else {
            ensureFloatImageBuffer(std::max(w1, w2), std::max(h1, h2));

            copyU8ToFloatAndUpload(static_cast<const uint8_t*>(buf1.ptr), w1, h1, d_img_float, whp1);
            data1.num_pts = 0;
            akazer.detectAndCompute(d_img_float, data1, whp1, true);

            bool diff_size = (w2 != w1 || h2 != h1);
            if (diff_size) {
                initIfNeeded(whp2);
                needs_init = true;
            }

            copyU8ToFloatAndUpload(static_cast<const uint8_t*>(buf2.ptr), w2, h2, d_img_float, whp2);
            data2.num_pts = 0;
            akazer.detectAndCompute(d_img_float, data2, whp2, true);
        }

        if (w2 != w1 || h2 != h1) {
            needs_init = true;
        }

        akaze::cuMatch(data1, data2);

        int n1 = data1.num_pts;
        bool apply_nndr = (nndr > 0.0f && nndr < 1.0f);

        std::vector<int> good_src;
        std::vector<int> good_dst;
        good_src.reserve(n1);
        good_dst.reserve(n1);
        for (int i = 0; i < n1; i++) {
            int midx = data1.h_data[i].match;
            if (midx < 0) continue;
            if (apply_nndr) {
                int d1s = data1.h_data[i].distance;
                int d2s = data1.h_data[i].distance2;
                if (d2s <= 0 || (float)d1s >= nndr * (float)d2s) continue;
            }
            good_src.push_back(i);
            good_dst.push_back(midx);
        }

        int n_good = (int)good_src.size();
        py::array_t<float> pts0({n_good, 2});
        py::array_t<float> pts1({n_good, 2});
        float* p0 = pts0.mutable_data();
        float* p1 = pts1.mutable_data();
        for (int i = 0; i < n_good; i++) {
            p0[i * 2 + 0] = data1.h_data[good_src[i]].x;
            p0[i * 2 + 1] = data1.h_data[good_src[i]].y;
            p1[i * 2 + 0] = data2.h_data[good_dst[i]].x;
            p1[i * 2 + 1] = data2.h_data[good_dst[i]].y;
        }

        return py::make_tuple(pts0, pts1);
    }
};


PYBIND11_MODULE(_cuda_akaze, m) {
    m.doc() = "CUDA-accelerated AKAZE feature detection and matching";

    m.def("init_device", [](int device_id) {
        return initDevice(device_id);
    }, py::arg("device_id") = 0);

    py::class_<AkazerPipeline>(m, "AkazerPipeline")
        .def(py::init<int>(), py::arg("max_pts") = 10000)
        .def("detect_and_match", &AkazerPipeline::detectAndMatch,
             py::arg("image1"), py::arg("image2"), py::arg("nndr") = 1.0f,
             py::arg("mode") = "fast");
}
