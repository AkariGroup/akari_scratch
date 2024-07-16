import contextlib
import pathlib
import sys
import cv2
import depthai as dai
import numpy
import time
import json
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel
from .detection_type import DetectionResult

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[2]
WEIGHT_PATH = (
    PACKAGE_DIR
    / "data/object_detection/yolov4_tiny_coco_416x416_openvino_2021.4_6shave.blob"
)
LABEL_PATH = PACKAGE_DIR / "data/object_detection/yolov4-tiny.json"
WIDTH = 416
HEIGHT = 416
PREV_WIDTH = 1920
PREV_HEIGHT = 1080
OUTPUT_WIDTH =480
OUTPUT_HEIGHT = 360

with LABEL_PATH.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})
# sync outputs
syncNN = True


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = numpy.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (numpy.clip(numpy.array(bbox), 0, 1) * normVals).astype(int)


def _render_frame(
    name: str, frame: numpy.ndarray, detections: Any
) -> Tuple[numpy.ndarray, List[DetectionResult]]:
    results = []
    id = 0
    # Resize the frame to crop mergin
    width = int(frame.shape[1] * 3 / 4)
    brank_width = frame.shape[1] - width
    height = int(frame.shape[0] * 9 / 16)
    brank_height = frame.shape[0] - height
    crop_frame = frame[
        int(brank_height / 2): int(frame.shape[0] - brank_height / 2),
        int(brank_width / 2): int(frame.shape[1] - brank_width / 2)
        ]
    output_frame = cv2.resize(crop_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    for detection in detections:
        detection.ymin = (detection.ymin - (brank_height / 2 / frame.shape[1]))
        detection.ymax =  (detection.ymax - (brank_height / 2 / frame.shape[1]))
        detection.xmin = (detection.xmin - (brank_width / 2 / frame.shape[1]))
        detection.xmax =  (detection.xmax - (brank_width / 2 / frame.shape[1]))

        origin_bbox = frameNorm(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )

        crop_h, crop_w = crop_frame.shape[:2]
        x_scale = OUTPUT_WIDTH / crop_w
        y_scale = OUTPUT_HEIGHT / crop_h
        bbox = [
            int(origin_bbox[0] * x_scale),
            int(origin_bbox[1] * y_scale),
            int(origin_bbox[2] * x_scale),
            int(origin_bbox[3] * y_scale)
        ]

        if not bbox[2] <= 0:
            if not bbox[0] > OUTPUT_WIDTH:
                cv2.putText(
                    output_frame,
                    labels[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    output_frame,
                    f"{int(detection.confidence * 100)}%",
                    (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                RED = (255, 0, 0)
                cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), RED, 2)
                result = DetectionResult(
                    id=id,
                    name=labels[detection.label],
                    x=bbox[0],
                    width=bbox[2] - bbox[0],
                    y=bbox[1],
                    height=bbox[3] - bbox[1],
                )
                results.append(result)
                id += 1
    # Show the output_frame
    return output_frame, results


class ObjectDetectionCapture:
    @staticmethod
    def _create_pipeline() -> dai.Pipeline:
        # Create pipeline
        pipeline = dai.Pipeline()
        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")
        # Properties
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setPreviewSize(PREV_WIDTH, PREV_HEIGHT)
        camRgb.setFps(10)
        xoutIsp = pipeline.create(dai.node.XLinkOut)
        xoutIsp.setStreamName("isp")
        camRgb.isp.link(xoutIsp.input)
        # Use ImageMqnip to resize with letterboxing
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(WIDTH * HEIGHT * 3)
        manip.initialConfig.setResizeThumbnail(WIDTH, HEIGHT)
        camRgb.preview.link(manip.inputImage)
        # Network specific settings
        detectionNetwork.setConfidenceThreshold(confidenceThreshold)
        detectionNetwork.setNumClasses(classes)
        detectionNetwork.setCoordinateSize(coordinates)
        detectionNetwork.setAnchors(anchors)
        detectionNetwork.setAnchorMasks(anchorMasks)
        detectionNetwork.setIouThreshold(iouThreshold)
        detectionNetwork.setBlobPath(WEIGHT_PATH)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)
        # Linking
        manip.out.link(detectionNetwork.input)
        detectionNetwork.passthrough.link(xoutRgb.input)
        detectionNetwork.out.link(nnOut.input)
        return pipeline

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()
        self._pipeline = ObjectDetectionCapture._create_pipeline()
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        self._rgb_queue = self._device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )
        self._isp_queue = self._device.getOutputQueue(name="isp")
        self._det_queue = self._device.getOutputQueue(
            name="nn", maxSize=4, blocking=False
        )
        frame = None
        detections = []
        self.detection_results = []

    def get_frame(self) -> Optional[numpy.ndarray]:
        inRgb = self._rgb_queue.get()
        inIsp = self._isp_queue.get()
        inDet = self._det_queue.get()
        if inRgb is not None:
            frame = inRgb.getCvFrame()
        if inDet is not None:
            detections = inDet.detections
        if frame is None:
            return None
        res_frame, self.detection_results = _render_frame("rgb", frame, detections)
        return res_frame

    def get_object_result(self) -> List[DetectionResult]:
        return self.detection_results

    def close(self) -> None:
        self._stack.close()
        self._rgb_queue = None
        self._detection_queue = None
