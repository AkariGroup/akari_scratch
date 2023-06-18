FROM ubuntu:22.04


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

COPY akari_scratch_server /resources/akari_scratch_server
COPY docker/requirements.txt /resources/requirements.txt
RUN pip install  -r /resources/requirements.txt
COPY config/akari /etc/akari

EXPOSE 52001
CMD ["python3", "-m", "akari_scratch_server.cli"]
