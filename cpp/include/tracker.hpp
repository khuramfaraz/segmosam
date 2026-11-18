#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class TrackerEngine {
public:
    TrackerEngine(const std::string& onnx_model_path = "", int keyframe_interval = 10);
    ~TrackerEngine() = default;

    // Initializes tracking on the first frame with bounding box (x, y, w, h)
    void init_target(const cv::Mat& frame, const cv::Rect& init_box);

    // Processes subsequent frames: runs ONNX on keyframes, Optical Flow in between
    cv::Mat process_frame(const cv::Mat& frame, double& latency_ms);

private:
    cv::Mat run_onnx_segmentation(const cv::Mat& frame, const cv::Rect& prompt_box);
    cv::Mat propagate_optical_flow(const cv::Mat& prev_gray, const cv::Mat& curr_gray);

    int keyframe_interval_;
    int frame_count_ = 0;

    cv::Mat prev_gray_;
    cv::Mat current_mask_;
    cv::Rect current_box_;
    std::vector<cv::Point2f> tracked_points_;

};
