FROM ubuntu:22.04


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip \
    && apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install  -r requirements.txt

COPY akari_scratch_server /resources/akari_scratch_server
RUN pip install  -e /resources/akari_scratch_server


EXPOSE 52001
CMD ["python3", "-m", "akari_scratch_server.cli"]
