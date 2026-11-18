#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tracker.hpp"

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 3 || buf.shape.at(2) != 3) {
        throw std::runtime_error("Expected HxWx3 uint8 RGB/BGR image");
    }
    return cv::Mat(
        static_cast<int>(buf.shape.at(0)),
        static_cast<int>(buf.shape.at(1)),
        CV_8UC3,
        buf.ptr
    ).clone();
}

py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    py::array_t<uint8_t> result({mat.rows, mat.cols});
    std::memcpy(result.mutable_data(), mat.data, mat.rows * mat.cols * sizeof(uint8_t));
    return result;
}

PYBIND11_MODULE(segmosam, m) {
    m.doc() = "segmosam: C++17 ONNX + Optical Flow Hybrid Video Segmentation Tracker";

    py::class_<TrackerEngine>(m, "TrackerEngine")
        .def(py::init<const std::string&, int>(),
             py::arg("onnx_model_path") = "",
             py::arg("keyframe_interval") = 10)
        .def("init_target", [](TrackerEngine& self, py::array_t<uint8_t> frame, const std::vector<int>& box) {
            if (box.size() != 4) {
                throw std::runtime_error("Box must contain 4 integers: x, y, w, h");
            }
            self.init_target(numpy_to_mat(frame), cv::Rect(box.at(0), box.at(1), box.at(2), box.at(3)));
        }, py::arg("frame"), py::arg("box"))
        .def("process_frame", [](TrackerEngine& self, py::array_t<uint8_t> frame) {
            double latency_ms = 0.0;
            cv::Mat mask = self.process_frame(numpy_to_mat(frame), latency_ms);
            return py::make_tuple(mat_to_numpy(mask), latency_ms);
        }, py::arg("frame"));
}
