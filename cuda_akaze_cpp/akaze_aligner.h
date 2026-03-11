#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace akaze {

struct AkazeAlignerConfig {
    float ransac_thresh = 2.5f;
    int ransac_max_iters = 2000;
    float ransac_confidence = 0.995f;
    float nndr = 0.8f;
    int max_pts = 10000;
    std::string mode = "fast";  // "fast" or "accurate"
};

/**
 * C++ AkazeAligner: GPU (CUDA) or CPU (OpenCV) AKAZE detect+match + cv::findHomography (RANSAC).
 * Matches Python cuda_akaze.AkazeAligner behavior.
 * device >= 0: CUDA device ID; device < 0 (e.g. -1): use OpenCV AKAZE on CPU.
 */
class AkazeAligner {
public:
    explicit AkazeAligner(int device = 0, const AkazeAlignerConfig& config = {});

    ~AkazeAligner();

    AkazeAligner(const AkazeAligner&) = delete;
    AkazeAligner& operator=(const AkazeAligner&) = delete;

    /** Single pair: compute H (3x3) and motion (H[0][2], H[1][2]). Returns false if insufficient matches. */
    bool findTransform(const cv::Mat& template_img, const cv::Mat& image,
                       cv::Mat& H_out, cv::Vec2f& motion_out);

    /** Batch: process B pairs, append H and motion to output vectors. */
    void findTransformBatch(const std::vector<cv::Mat>& templates,
                            const std::vector<cv::Mat>& images,
                            std::vector<cv::Mat>& H_out,
                            std::vector<cv::Vec2f>& motion_out);

private:
    AkazeAlignerConfig config_;
    int device_;
    struct Impl;
    Impl* impl_ = nullptr;

    void detectAndMatchCpu(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                           std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);
};

}  // namespace akaze
