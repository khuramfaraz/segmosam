FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/app/build/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "${VIRTUAL_ENV}"

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

RUN cmake -S . -B build \
        -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build --parallel

CMD ["python", "tests/smoke_test.py"]
