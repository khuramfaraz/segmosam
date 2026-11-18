# segmosam

`segmosam` is a C++17 tracking extension exposed to Python through pybind11. The current segmentation routine is a placeholder that returns the initialized bounding box as a mask; ONNX model inference is not wired in yet.

## Local Build

Install the native build dependencies. On Fedora:

```sh
sudo dnf install -y cmake gcc-c++ opencv-devel python3-devel
```

On Ubuntu or Debian:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev python3-dev python3-venv
```

Create a venv if you do not already have one, then install the Python build/test dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure and build the extension using that venv:

```sh
cmake -S . -B build \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build --parallel
```

Run the API smoke test:

```sh
PYTHONPATH="$PWD/build/python" .venv/bin/python tests/smoke_test.py
```

The test checks that the extension imports, accepts a synthetic frame and bounding box, and returns the expected 2-D `uint8` mask.

## Docker

Build and run the CPU image:

```sh
docker build -t segmosam:dev .
docker run --rm segmosam:dev
```

The image builds the extension and runs the same smoke test as its default command.