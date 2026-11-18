#include "tracker.hpp"
#include <chrono>
#include <algorithm>

TrackerEngine::TrackerEngine(const std::string&, int keyframe_interval)
    : keyframe_interval_(keyframe_interval) {}

void TrackerEngine::init_target(const cv::Mat& frame, const cv::Rect& init_box) {
    cv::cvtColor(frame, prev_gray_, cv::COLOR_BGR2GRAY);
    current_box_ = init_box;
    frame_count_ = 0;

    current_mask_ = run_onnx_segmentation(frame, current_box_);
    cv::goodFeaturesToTrack(prev_gray_, tracked_points_, 80, 0.01, 5, current_mask_);
}

cv::Mat TrackerEngine::process_frame(const cv::Mat& frame, double& latency_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat curr_gray;
    cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
    frame_count_++;

    if (frame_count_ % keyframe_interval_ == 0 || tracked_points_.size() < 10) {
        current_mask_ = run_onnx_segmentation(frame, current_box_);
        cv::goodFeaturesToTrack(curr_gray, tracked_points_, 80, 0.01, 5, current_mask_);
    } else {
        current_mask_ = propagate_optical_flow(prev_gray_, curr_gray);
    }

    prev_gray_ = curr_gray.clone();

    auto end = std::chrono::high_resolution_clock::now();
    latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return current_mask_;
}

cv::Mat TrackerEngine::run_onnx_segmentation(const cv::Mat& frame, const cv::Rect& prompt_box) {
    // Placeholder box mask until MobileSAM/SAM2 ONNX tensor inference is wired in
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Rect safe_box = prompt_box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe_box.area() > 0) {
        mask(safe_box).setTo(255);
    }
    return mask;
}

cv::Mat TrackerEngine::propagate_optical_flow(const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
    if (tracked_points_.empty()) {
        return current_mask_;
    }

    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, tracked_points_, next_points, status, err);

    std::vector<float> dxs, dys;
    std::vector<cv::Point2f> valid_next;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status.at(i)) {
            dxs.push_back(next_points.at(i).x - tracked_points_.at(i).x);
            dys.push_back(next_points.at(i).y - tracked_points_.at(i).y);
            valid_next.push_back(next_points.at(i));
        }
    }

    if (!dxs.empty()) {
        std::nth_element(dxs.begin(), dxs.begin() + dxs.size() / 2, dxs.end());
        std::nth_element(dys.begin(), dys.begin() + dys.size() / 2, dys.end());
        float median_dx = dxs.at(dxs.size() / 2);
        float median_dy = dys.at(dys.size() / 2);

        cv::Mat warp_mat = (cv::Mat_<double>(2, 3) << 1, 0, median_dx, 0, 1, median_dy);
        cv::Mat warped_mask;
        cv::warpAffine(current_mask_, warped_mask, warp_mat, current_mask_.size(), cv::INTER_NEAREST);
        current_mask_ = warped_mask;

        current_box_.x += static_cast<int>(median_dx);
        current_box_.y += static_cast<int>(median_dy);
    }

    tracked_points_ = valid_next;
    return current_mask_;
}
