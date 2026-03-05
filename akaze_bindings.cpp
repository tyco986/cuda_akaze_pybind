#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "akaze.h"
#include "cuda_utils.h"
#include <stdexcept>
#include <vector>

namespace py = pybind11;

#define AKAZE_FLEN 61

void set_device(int device_id) {
    initDevice(device_id);
}

py::dict detect_and_compute(
    py::array img,
    bool use_fast = true,
    int noctaves = 4,
    int max_scale = 4,
    float per = 0.7f,
    float kcontrast = 0.03f,
    float soffset = 1.6f,
    bool reordering = true,
    float derivative_factor = 1.5f,
    float dthreshold = 0.001f,
    int diffusivity = 1,
    int descriptor_pattern_size = 10,
    int max_pts = 10000
) {
    auto buf = img.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be 2D (H, W)");
    }
    int h = buf.shape[0];
    int w = buf.shape[1];

    int3 whp;
    whp.x = w;
    whp.y = h;
    whp.z = iAlignUp(w, 128);

    akaze::AkazeData akaze_data;
    akaze::initAkazeData(akaze_data, max_pts, true, true);

    akaze::Akazer detector;
    detector.init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
        derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

    if (use_fast) {
        if (buf.format != py::format_descriptor<uint8_t>::format()) {
            throw std::runtime_error("Fast mode requires uint8 image. Use img.astype(np.uint8)");
        }
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        ssize_t spitch = buf.strides[0];
        if (spitch != static_cast<ssize_t>(w)) {
            throw std::runtime_error("Image must be contiguous");
        }

        size_t dpitch = sizeof(uint8_t) * whp.z;
        unsigned char* d_img = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&d_img, &tmp_pitch, sizeof(uint8_t) * w, h));
        CHECK(cudaMemcpy2D(d_img, dpitch, ptr, spitch, spitch, h, cudaMemcpyHostToDevice));

        detector.fastDetectAndCompute(d_img, akaze_data, whp, true);

        CHECK(cudaFree(d_img));
    } else {
        if (buf.format != py::format_descriptor<float>::format()) {
            throw std::runtime_error("Standard mode requires float32 [0,1]. Use (img/255.0).astype(np.float32)");
        }
        float* ptr = static_cast<float*>(buf.ptr);
        ssize_t spitch = buf.strides[0];
        if (spitch != static_cast<ssize_t>(w * sizeof(float))) {
            throw std::runtime_error("Image must be contiguous");
        }

        size_t dpitch = sizeof(float) * whp.z;
        float* d_img = nullptr;
        size_t tmp_pitch = 0;
        CHECK(cudaMallocPitch((void**)&d_img, &tmp_pitch, sizeof(float) * w, h));
        CHECK(cudaMemcpy2D(d_img, dpitch, ptr, spitch, spitch, h, cudaMemcpyHostToDevice));

        detector.detectAndCompute(d_img, akaze_data, whp, true);

        CHECK(cudaFree(d_img));
    }

    int n = akaze_data.num_pts;
    py::array_t<float> keypoints({n, 6});
    py::array_t<uint8_t> descriptors({n, AKAZE_FLEN});

    auto kp_acc = keypoints.mutable_unchecked<2>();
    auto desc_acc = descriptors.mutable_unchecked<2>();

    akaze::AkazePoint* pts = akaze_data.h_data;
    for (int i = 0; i < n; i++) {
        kp_acc(i, 0) = pts[i].x;
        kp_acc(i, 1) = pts[i].y;
        kp_acc(i, 2) = static_cast<float>(pts[i].octave);
        kp_acc(i, 3) = pts[i].response;
        kp_acc(i, 4) = pts[i].size;
        kp_acc(i, 5) = pts[i].angle;
        for (int j = 0; j < AKAZE_FLEN; j++) {
            desc_acc(i, j) = pts[i].features[j];
        }
    }

    akaze::freeAkazeData(akaze_data);

    py::dict result;
    result["keypoints"] = keypoints;
    result["descriptors"] = descriptors;
    return result;
}

py::dict match(
    py::array img1,
    py::array img2,
    bool use_fast = true,
    int noctaves = 4,
    int max_scale = 4,
    float per = 0.7f,
    float kcontrast = 0.03f,
    float soffset = 1.6f,
    bool reordering = true,
    float derivative_factor = 1.5f,
    float dthreshold = 0.001f,
    int diffusivity = 1,
    int descriptor_pattern_size = 10,
    int max_pts = 10000
) {
    auto buf1 = img1.request();
    auto buf2 = img2.request();
    if (buf1.ndim != 2 || buf2.ndim != 2) {
        throw std::runtime_error("Images must be 2D (H, W)");
    }
    int h1 = buf1.shape[0], w1 = buf1.shape[1];
    int h2 = buf2.shape[0], w2 = buf2.shape[1];

    int3 whp1, whp2;
    whp1.x = w1; whp1.y = h1; whp1.z = iAlignUp(w1, 128);
    whp2.x = w2; whp2.y = h2; whp2.z = iAlignUp(w2, 128);

    akaze::AkazeData akaze_data1, akaze_data2;
    akaze::initAkazeData(akaze_data1, max_pts, true, true);
    akaze::initAkazeData(akaze_data2, max_pts, true, true);

    akaze::Akazer detector;
    detector.init(whp1, noctaves, max_scale, per, kcontrast, soffset, reordering,
        derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);

    auto run_detect = [&](void* ptr, ssize_t spitch, int h, int w, int3 whp, akaze::AkazeData& data) {
        if (use_fast) {
            size_t dp = sizeof(uint8_t) * whp.z;
            unsigned char* d_img = nullptr;
            size_t tp = 0;
            CHECK(cudaMallocPitch((void**)&d_img, &tp, sizeof(uint8_t) * w, h));
            CHECK(cudaMemcpy2D(d_img, dp, ptr, spitch, spitch, h, cudaMemcpyHostToDevice));
            detector.fastDetectAndCompute(d_img, data, whp, true);
            CHECK(cudaFree(d_img));
        } else {
            size_t dp = sizeof(float) * whp.z;
            float* d_img = nullptr;
            size_t tp = 0;
            CHECK(cudaMallocPitch((void**)&d_img, &tp, sizeof(float) * w, h));
            CHECK(cudaMemcpy2D(d_img, dp, ptr, spitch, spitch, h, cudaMemcpyHostToDevice));
            detector.detectAndCompute(d_img, data, whp, true);
            CHECK(cudaFree(d_img));
        }
    };

    run_detect(buf1.ptr, buf1.strides[0], h1, w1, whp1, akaze_data1);
    run_detect(buf2.ptr, buf2.strides[0], h2, w2, whp2, akaze_data2);

    akaze::cuMatch(akaze_data1, akaze_data2);

    int n1 = akaze_data1.num_pts;
    int n2 = akaze_data2.num_pts;

    py::array_t<float> kp1({n1, 6});
    py::array_t<float> kp2({n2, 6});
    py::array_t<uint8_t> desc1({n1, AKAZE_FLEN});
    py::array_t<uint8_t> desc2({n2, AKAZE_FLEN});

    auto kp1_acc = kp1.mutable_unchecked<2>();
    auto kp2_acc = kp2.mutable_unchecked<2>();
    auto d1_acc = desc1.mutable_unchecked<2>();
    auto d2_acc = desc2.mutable_unchecked<2>();

    akaze::AkazePoint* pts1 = akaze_data1.h_data;
    akaze::AkazePoint* pts2 = akaze_data2.h_data;
    for (int i = 0; i < n1; i++) {
        kp1_acc(i, 0) = pts1[i].x;
        kp1_acc(i, 1) = pts1[i].y;
        kp1_acc(i, 2) = static_cast<float>(pts1[i].octave);
        kp1_acc(i, 3) = pts1[i].response;
        kp1_acc(i, 4) = pts1[i].size;
        kp1_acc(i, 5) = pts1[i].angle;
        for (int j = 0; j < AKAZE_FLEN; j++) d1_acc(i, j) = pts1[i].features[j];
    }
    for (int i = 0; i < n2; i++) {
        kp2_acc(i, 0) = pts2[i].x;
        kp2_acc(i, 1) = pts2[i].y;
        kp2_acc(i, 2) = static_cast<float>(pts2[i].octave);
        kp2_acc(i, 3) = pts2[i].response;
        kp2_acc(i, 4) = pts2[i].size;
        kp2_acc(i, 5) = pts2[i].angle;
        for (int j = 0; j < AKAZE_FLEN; j++) d2_acc(i, j) = pts2[i].features[j];
    }

    std::vector<py::tuple> matches;
    for (int i = 0; i < n1; i++) {
        int k = pts1[i].match;
        if (k >= 0) {
            matches.push_back(py::make_tuple(i, k, pts1[i].distance));
        }
    }

    akaze::freeAkazeData(akaze_data1);
    akaze::freeAkazeData(akaze_data2);

    py::dict result;
    result["keypoints1"] = kp1;
    result["keypoints2"] = kp2;
    result["descriptors1"] = desc1;
    result["descriptors2"] = desc2;
    result["matches"] = matches;
    return result;
}

PYBIND11_MODULE(_cuda_akaze, m) {
    m.doc() = "CUDA-AKAZE: keypoint detection, descriptor computation, and matching";
    m.def("set_device", &set_device, py::arg("device_id") = 0,
        "Set CUDA device (optional, default 0)");
    m.def("detect_and_compute", &detect_and_compute,
        py::arg("img"),
        py::arg("use_fast") = true,
        py::arg("noctaves") = 4,
        py::arg("max_scale") = 4,
        py::arg("per") = 0.7f,
        py::arg("kcontrast") = 0.03f,
        py::arg("soffset") = 1.6f,
        py::arg("reordering") = true,
        py::arg("derivative_factor") = 1.5f,
        py::arg("dthreshold") = 0.001f,
        py::arg("diffusivity") = 1,
        py::arg("descriptor_pattern_size") = 10,
        py::arg("max_pts") = 10000,
        "Detect AKAZE keypoints and compute descriptors.");
    m.def("match", &match,
        py::arg("img1"),
        py::arg("img2"),
        py::arg("use_fast") = true,
        py::arg("noctaves") = 4,
        py::arg("max_scale") = 4,
        py::arg("per") = 0.7f,
        py::arg("kcontrast") = 0.03f,
        py::arg("soffset") = 1.6f,
        py::arg("reordering") = true,
        py::arg("derivative_factor") = 1.5f,
        py::arg("dthreshold") = 0.001f,
        py::arg("diffusivity") = 1,
        py::arg("descriptor_pattern_size") = 10,
        py::arg("max_pts") = 10000,
        "Detect, compute, and match two images.");
}
