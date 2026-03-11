#include "akaze_aligner.h"
#include "akaze.h"
#include "cuda_utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>

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

bool parseMode(const std::string& mode) {
    if (mode == "fast") return true;
    if (mode == "accurate") return false;
    return true;  // default fast
}

}  // namespace

namespace akaze {

struct AkazeAligner::Impl {
    akaze::Akazer akazer;
    akaze::AkazeData data1{}, data2{};
    uint8_t* d_img = nullptr;
    float* d_img_float = nullptr;
    size_t img_pitch = 0;
    size_t float_img_pitch = 0;
    int buf_width = 0, buf_height = 0;
    int max_pts;
    bool use_fast;
    bool needs_init = true;
    int c_width = 0, c_height = 0;

    Impl(int _max_pts, bool _use_fast) : max_pts(_max_pts), use_fast(_use_fast) {
        std::memset(&data1, 0, sizeof(data1));
        std::memset(&data2, 0, sizeof(data2));
        akaze::initAkazeData(data1, max_pts, true, true);
        akaze::initAkazeData(data2, max_pts, true, true);
    }

    ~Impl() {
        if (d_img) cudaFree(d_img);
        if (d_img_float) cudaFree(d_img_float);
        akaze::freeAkazeData(data1);
        akaze::freeAkazeData(data2);
    }

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

    void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                        std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
        if (img1.empty() || img2.empty() || img1.type() != CV_8UC1 || img2.type() != CV_8UC1) {
            pts1.clear();
            pts2.clear();
            return;
        }
        int w1 = img1.cols, h1 = img1.rows;
        int w2 = img2.cols, h2 = img2.rows;
        int3 whp1{w1, h1, iAlignUp(w1, 128)};
        int3 whp2{w2, h2, iAlignUp(w2, 128)};

        initIfNeeded(whp1);

        if (use_fast) {
            ensureImageBuffer(std::max(w1, w2), std::max(h1, h2));

            size_t spitch1 = sizeof(uint8_t) * w1;
            CHECK(cudaMemcpy2D(d_img, img_pitch, img1.data, spitch1, spitch1, h1, cudaMemcpyHostToDevice));
            data1.num_pts = 0;
            akazer.fastDetectAndCompute(d_img, data1, whp1, true);

            bool diff_size = (w2 != w1 || h2 != h1);
            if (diff_size) {
                initIfNeeded(whp2);
                needs_init = true;
            }

            size_t spitch2 = sizeof(uint8_t) * w2;
            CHECK(cudaMemcpy2D(d_img, img_pitch, img2.data, spitch2, spitch2, h2, cudaMemcpyHostToDevice));
            data2.num_pts = 0;
            akazer.fastDetectAndCompute(d_img, data2, whp2, true);
        } else {
            ensureFloatImageBuffer(std::max(w1, w2), std::max(h1, h2));

            copyU8ToFloatAndUpload(img1.data, w1, h1, d_img_float, whp1);
            data1.num_pts = 0;
            akazer.detectAndCompute(d_img_float, data1, whp1, true);

            bool diff_size = (w2 != w1 || h2 != h1);
            if (diff_size) {
                initIfNeeded(whp2);
                needs_init = true;
            }

            copyU8ToFloatAndUpload(img2.data, w2, h2, d_img_float, whp2);
            data2.num_pts = 0;
            akazer.detectAndCompute(d_img_float, data2, whp2, true);
        }

        if (w2 != w1 || h2 != h1) {
            needs_init = true;
        }

        akaze::cuMatch(data1, data2);

        pts1.clear();
        pts2.clear();
        bool apply_nndr = (nndr > 0.0f && nndr < 1.0f);
        for (int i = 0; i < data1.num_pts; i++) {
            int midx = data1.h_data[i].match;
            if (midx < 0) continue;
            if (apply_nndr) {
                int d1s = data1.h_data[i].distance;
                int d2s = data1.h_data[i].distance2;
                if (d2s <= 0 || (float)d1s >= nndr * (float)d2s) continue;
            }
            pts1.push_back(cv::Point2f(data1.h_data[i].x, data1.h_data[i].y));
            pts2.push_back(cv::Point2f(data2.h_data[midx].x, data2.h_data[midx].y));
        }
    }
};

AkazeAligner::AkazeAligner(int device, const AkazeAlignerConfig& config)
    : config_(config), device_(device) {
    if (device >= 0) {
        initDevice(device);
        impl_ = new Impl(config_.max_pts, parseMode(config_.mode));
    } else {
        impl_ = nullptr;
    }
}

AkazeAligner::~AkazeAligner() {
    if (impl_) delete impl_;
}

void AkazeAligner::detectAndMatchCpu(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                                     std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
    pts1.clear();
    pts2.clear();
    if (img1.empty() || img2.empty() || img1.type() != CV_8UC1 || img2.type() != CV_8UC1)
        return;
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 4, 4, cv::KAZE::DIFF_PM_G2);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    if (desc1.empty() || desc2.empty() || kp1.empty() || kp2.empty())
        return;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    if (nndr >= 1.0f) {
        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);
        for (const auto& m : matches) {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }
    } else {
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc1, desc2, knn, 2);
        for (const auto& m_n : knn) {
            if (m_n.size() == 2 && m_n[0].distance < nndr * m_n[1].distance) {
                pts1.push_back(kp1[m_n[0].queryIdx].pt);
                pts2.push_back(kp2[m_n[0].trainIdx].pt);
            } else if (m_n.size() == 1) {
                pts1.push_back(kp1[m_n[0].queryIdx].pt);
                pts2.push_back(kp2[m_n[0].trainIdx].pt);
            }
        }
    }
}

bool AkazeAligner::findTransform(const cv::Mat& template_img, const cv::Mat& image,
                                 cv::Mat& H_out, cv::Vec2f& motion_out) {
    cv::Mat tpl = template_img;
    cv::Mat img = image;
    if (template_img.channels() == 3)
        cv::cvtColor(template_img, tpl, cv::COLOR_BGR2GRAY);
    if (image.channels() == 3)
        cv::cvtColor(image, img, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts1, pts2;
    if (impl_) {
        impl_->detectAndMatch(tpl, img, config_.nndr, pts1, pts2);
    } else {
        detectAndMatchCpu(tpl, img, config_.nndr, pts1, pts2);
    }

    if (pts1.size() < 4) {
        H_out = cv::Mat::eye(3, 3, CV_32F);
        motion_out = cv::Vec2f(0, 0);
        return false;
    }

    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, config_.ransac_thresh,
                                   cv::noArray(), config_.ransac_max_iters,
                                   config_.ransac_confidence);
    if (H.empty()) {
        H_out = cv::Mat::eye(3, 3, CV_32F);
        motion_out = cv::Vec2f(0, 0);
        return false;
    }
    H.convertTo(H_out, CV_32F);
    motion_out = cv::Vec2f(H_out.at<float>(0, 2), H_out.at<float>(1, 2));
    return true;
}

void AkazeAligner::findTransformBatch(const std::vector<cv::Mat>& templates,
                                     const std::vector<cv::Mat>& images,
                                     std::vector<cv::Mat>& H_out,
                                     std::vector<cv::Vec2f>& motion_out) {
    const size_t B = std::min(templates.size(), images.size());
    H_out.reserve(H_out.size() + B);
    motion_out.reserve(motion_out.size() + B);
    for (size_t b = 0; b < B; b++) {
        cv::Mat H;
        cv::Vec2f mot;
        findTransform(templates[b], images[b], H, mot);
        H_out.push_back(H);
        motion_out.push_back(mot);
    }
}

}  // namespace akaze
