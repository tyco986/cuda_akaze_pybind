#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "akaze.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_cuda_akaze, m) {
    m.doc() = "CUDA-accelerated AKAZE feature detection and matching";

    py::class_<akaze::Akazer>(m, "Akazer")
        .def(py::init<>())
        .def("detect_and_compute", [](akaze::Akazer& self, py::array_t<float> image, bool compute_descriptor,
                int noctaves, int max_scale, float per, float kcontrast, float soffset, bool reordering,
                float derivative_factor, float dthreshold, int diffusivity, int descriptor_pattern_size) {
            py::buffer_info buf = image.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Image must be 2D (height, width)");
            }
            if (buf.itemsize != sizeof(float)) {
                throw std::runtime_error("Image must be float32");
            }
            int height = buf.shape[0];
            int width = buf.shape[1];
            size_t pitch = sizeof(float) * (size_t)iAlignUp(width, 128);

            float* d_img = nullptr;
            cudaError_t err = cudaMallocPitch((void**)&d_img, &pitch, sizeof(float) * width, height);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMallocPitch failed: ") + cudaGetErrorString(err));
            }

            size_t spitch = sizeof(float) * width;
            err = cudaMemcpy2D(d_img, pitch, buf.ptr, spitch, spitch, height, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_img);
                throw std::runtime_error(std::string("cudaMemcpy2D failed: ") + cudaGetErrorString(err));
            }

            int3 whp;
            whp.x = width;
            whp.y = height;
            whp.z = iAlignUp(width, 128);

            akaze::AkazeData data;
            akaze::initAkazeData(data, 10000, true, true);
            self.init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
                      derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);
            self.detectAndCompute(d_img, data, whp, compute_descriptor);

            cudaFree(d_img);

            py::array_t<float> keypoints({data.num_pts, 6});
            py::array_t<uint8_t> descriptors;
            if (compute_descriptor) {
                descriptors = py::array_t<uint8_t>({data.num_pts, 61});
            }

            float* kp_ptr = keypoints.mutable_data();
            uint8_t* desc_ptr = compute_descriptor ? descriptors.mutable_data() : nullptr;

            for (int i = 0; i < data.num_pts; i++) {
                akaze::AkazePoint& p = data.h_data[i];
                kp_ptr[i * 6 + 0] = p.x;
                kp_ptr[i * 6 + 1] = p.y;
                kp_ptr[i * 6 + 2] = p.size;
                kp_ptr[i * 6 + 3] = p.angle;
                kp_ptr[i * 6 + 4] = p.response;
                kp_ptr[i * 6 + 5] = static_cast<float>(p.octave);
                if (desc_ptr) {
                    for (int j = 0; j < 61; j++) {
                        desc_ptr[i * 61 + j] = p.features[j];
                    }
                }
            }

            akaze::freeAkazeData(data);

            py::object desc_obj = compute_descriptor ? py::object(descriptors) : py::none();
            return py::make_tuple(keypoints, desc_obj);
        }, py::arg("image"), py::arg("compute_descriptor") = true,
           py::arg("noctaves") = 4, py::arg("max_scale") = 4, py::arg("per") = 0.7f, py::arg("kcontrast") = 0.03f,
           py::arg("soffset") = 1.6f, py::arg("reordering") = true, py::arg("derivative_factor") = 1.5f,
           py::arg("dthreshold") = 0.001f, py::arg("diffusivity") = 1, py::arg("descriptor_pattern_size") = 10)
        .def("fast_detect_and_compute", [](akaze::Akazer& self, py::array_t<uint8_t> image, bool compute_descriptor,
                int noctaves, int max_scale, float per, float kcontrast, float soffset, bool reordering,
                float derivative_factor, float dthreshold, int diffusivity, int descriptor_pattern_size) {
            py::buffer_info buf = image.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Image must be 2D (height, width)");
            }
            int height = buf.shape[0];
            int width = buf.shape[1];
            size_t pitch = sizeof(uint8_t) * (size_t)iAlignUp(width, 128);

            uint8_t* d_img = nullptr;
            cudaError_t err = cudaMallocPitch((void**)&d_img, &pitch, sizeof(uint8_t) * width, height);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMallocPitch failed: ") + cudaGetErrorString(err));
            }

            size_t spitch = sizeof(uint8_t) * width;
            err = cudaMemcpy2D(d_img, pitch, buf.ptr, spitch, spitch, height, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_img);
                throw std::runtime_error(std::string("cudaMemcpy2D failed: ") + cudaGetErrorString(err));
            }

            int3 whp;
            whp.x = width;
            whp.y = height;
            whp.z = iAlignUp(width, 128);

            akaze::AkazeData data;
            akaze::initAkazeData(data, 10000, true, true);
            self.init(whp, noctaves, max_scale, per, kcontrast, soffset, reordering,
                      derivative_factor, dthreshold, diffusivity, descriptor_pattern_size);
            self.fastDetectAndCompute(d_img, data, whp, compute_descriptor);

            cudaFree(d_img);

            py::array_t<float> keypoints({data.num_pts, 6});
            py::array_t<uint8_t> descriptors;
            if (compute_descriptor) {
                descriptors = py::array_t<uint8_t>({data.num_pts, 61});
            }

            float* kp_ptr = keypoints.mutable_data();
            uint8_t* desc_ptr = compute_descriptor ? descriptors.mutable_data() : nullptr;

            for (int i = 0; i < data.num_pts; i++) {
                akaze::AkazePoint& p = data.h_data[i];
                kp_ptr[i * 6 + 0] = p.x;
                kp_ptr[i * 6 + 1] = p.y;
                kp_ptr[i * 6 + 2] = p.size;
                kp_ptr[i * 6 + 3] = p.angle;
                kp_ptr[i * 6 + 4] = p.response;
                kp_ptr[i * 6 + 5] = static_cast<float>(p.octave);
                if (desc_ptr) {
                    for (int j = 0; j < 61; j++) {
                        desc_ptr[i * 61 + j] = p.features[j];
                    }
                }
            }

            akaze::freeAkazeData(data);

            py::object desc_obj = compute_descriptor ? py::object(descriptors) : py::none();
            return py::make_tuple(keypoints, desc_obj);
        }, py::arg("image"), py::arg("compute_descriptor") = true,
           py::arg("noctaves") = 4, py::arg("max_scale") = 4, py::arg("per") = 0.7f, py::arg("kcontrast") = 0.03f,
           py::arg("soffset") = 1.6f, py::arg("reordering") = true, py::arg("derivative_factor") = 1.5f,
           py::arg("dthreshold") = 0.001f, py::arg("diffusivity") = 1, py::arg("descriptor_pattern_size") = 10);

    m.def("init_device", [](int device_id) {
        return initDevice(device_id);
    }, py::arg("device_id") = 0);

    m.def("match", [](py::array_t<float> kp1, py::array_t<uint8_t> desc1,
                      py::array_t<float> kp2, py::array_t<uint8_t> desc2) {
        py::buffer_info b1 = kp1.request(), b2 = kp2.request();
        py::buffer_info d1 = desc1.request(), d2 = desc2.request();
        if (b1.ndim != 2 || b2.ndim != 2 || b1.shape[1] != 6 || b2.shape[1] != 6) {
            throw std::runtime_error("Keypoints must be (N, 6)");
        }
        if (d1.shape[0] != b1.shape[0] || d2.shape[0] != b2.shape[0] || d1.shape[1] != 61 || d2.shape[1] != 61) {
            throw std::runtime_error("Descriptors must be (N, 61)");
        }
        int n1 = b1.shape[0], n2 = b2.shape[0];

        akaze::AkazeData data1, data2;
        akaze::initAkazeData(data1, n1, true, true);
        akaze::initAkazeData(data2, n2, true, true);
        data1.num_pts = n1;
        data2.num_pts = n2;

        float* kp1_ptr = static_cast<float*>(b1.ptr);
        float* kp2_ptr = static_cast<float*>(b2.ptr);
        uint8_t* d1_ptr = static_cast<uint8_t*>(d1.ptr);
        uint8_t* d2_ptr = static_cast<uint8_t*>(d2.ptr);

        for (int i = 0; i < n1; i++) {
            data1.h_data[i].x = kp1_ptr[i * 6 + 0];
            data1.h_data[i].y = kp1_ptr[i * 6 + 1];
            data1.h_data[i].size = kp1_ptr[i * 6 + 2];
            data1.h_data[i].angle = kp1_ptr[i * 6 + 3];
            data1.h_data[i].response = kp1_ptr[i * 6 + 4];
            data1.h_data[i].octave = static_cast<int>(kp1_ptr[i * 6 + 5]);
            for (int j = 0; j < 61; j++) data1.h_data[i].features[j] = d1_ptr[i * 61 + j];
        }
        for (int i = 0; i < n2; i++) {
            data2.h_data[i].x = kp2_ptr[i * 6 + 0];
            data2.h_data[i].y = kp2_ptr[i * 6 + 1];
            data2.h_data[i].size = kp2_ptr[i * 6 + 2];
            data2.h_data[i].angle = kp2_ptr[i * 6 + 3];
            data2.h_data[i].response = kp2_ptr[i * 6 + 4];
            data2.h_data[i].octave = static_cast<int>(kp2_ptr[i * 6 + 5]);
            for (int j = 0; j < 61; j++) data2.h_data[i].features[j] = d2_ptr[i * 61 + j];
        }

        CHECK(cudaMemcpy(data1.d_data, data1.h_data, n1 * sizeof(akaze::AkazePoint), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(data2.d_data, data2.h_data, n2 * sizeof(akaze::AkazePoint), cudaMemcpyHostToDevice));

        akaze::cuMatch(data1, data2);

        CHECK(cudaMemcpy2D(&data1.h_data[0].match, sizeof(akaze::AkazePoint),
                          &data1.d_data[0].match, sizeof(akaze::AkazePoint),
                          4 * sizeof(float), n1, cudaMemcpyDeviceToHost));

        py::array_t<int32_t> matches({n1, 2});
        int32_t* m_ptr = matches.mutable_data();
        for (int i = 0; i < n1; i++) {
            m_ptr[i * 2 + 0] = i;
            m_ptr[i * 2 + 1] = data1.h_data[i].match;
        }

        akaze::freeAkazeData(data1);
        akaze::freeAkazeData(data2);

        return matches;
    }, py::arg("keypoints1"), py::arg("descriptors1"),
       py::arg("keypoints2"), py::arg("descriptors2"));
}
