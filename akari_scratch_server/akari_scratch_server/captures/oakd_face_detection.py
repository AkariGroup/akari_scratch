import os
import time
import pathlib
import cv2
import contextlib
import depthai as dai
import numpy as np
from typing import Any, Optional, List, Tuple

from .detection_type import DetectionResult
from .utils.priorbox import PriorBox
from .utils.utils import draw

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[2]
WEIGHT_PATH = PACKAGE_DIR / "data/face_detection/face_detection_yunet_120x160.blob"


NN_WIDTH, NN_HEIGHT = 160, 120
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
TOP_K = 750
OUTPUT_WIDTH =480
OUTPUT_HEIGHT =360

def _render_frame(
    name: str, frame: np.ndarray, detections: Any
) -> Tuple[np.ndarray, List[int]]:
    # get all layers
    conf = np.array(detections.getLayerFp16("conf")).reshape((1076, 2))
    iou = np.array(detections.getLayerFp16("iou")).reshape((1076, 1))
    loc = np.array(detections.getLayerFp16("loc")).reshape((1076, 14))

    # decode
    pb = PriorBox(
        input_shape=(NN_WIDTH, NN_HEIGHT),
        output_shape=(frame.shape[1], frame.shape[0]),
    )
    dets = pb.decode(loc, conf, iou, CONFIDENCE_THRESHOLD)
    bbox = []
    # NMS
    if dets.shape[0] > 0:
        # NMS from OpenCV
        bboxes = dets[:, 0:4]
        scores = dets[:, -1]

        keep_idx = cv2.dnn.NMSBoxes(
            bboxes=bboxes.tolist(),
            scores=scores.tolist(),
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=IOU_THRESHOLD,
            eta=1,
            top_k=TOP_K,
        )  # returns [box_num, class_num]

        keep_idx = np.squeeze(keep_idx)  # [box_num, class_num] -> [box_num]
        dets = dets[keep_idx]
        if dets.ndim == 1:
            dets = np.expand_dims(dets, 0)
        frame = draw(
            img=frame,
            bboxes=dets[:, :4],
            landmarks=np.reshape(dets[:, 4:14], (-1, 5, 2)),
            scores=dets[:, -1],
        )
        bbox = bboxes[0]
    resized_frame: np.ndarray = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return resized_frame, bbox


class FaceDetectionCapture:
    @staticmethod
    def _create_pipeline() -> dai.Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(WEIGHT_PATH)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        cam.setInterleaved(False)
        cam.setFps(40)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        manip.inputConfig.setWaitForMessage(False)

        # Create outputs
        xout_cam = pipeline.create(dai.node.XLinkOut)
        xout_cam.setStreamName("cam")

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        cam.preview.link(manip.inputImage)
        cam.preview.link(xout_cam.input)
        manip.out.link(detection_nn.input)
        detection_nn.out.link(xout_nn.input)

        return pipeline

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()
        self._pipeline = FaceDetectionCapture._create_pipeline()
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        self._rgb_queue = self._device.getOutputQueue(name="cam", maxSize=4, blocking=False)  # type: ignore
        self._detection_queue = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)  # type: ignore
        self.bbox = []

    def get_frame(self) -> Optional[np.ndarray]:
        rgb_queue = self._rgb_queue
        detection_queue = self._detection_queue

        if rgb_queue is None or detection_queue is None:
            return None

        frame: Optional[np.ndarray] = rgb_queue.get().getCvFrame()  # type: ignore
        self.detection = detection_queue.get()
        if frame is None or self.detection is None:
            return None
        res_frame, self.bbox = _render_frame("rgb", frame, self.detection)
        return res_frame

    def get_face_result(self) -> List[DetectionResult]:
        result_list = []
        if (len(self.bbox)) > 0:
            result = DetectionResult(
            id = 0,
            name = "face",
            x= self.bbox[0],
            y  = self.bbox[1],
            width = self.bbox[2],
            height = self.bbox[3])
            result_list.append(result)
        return result_list

    def close(self) -> None:
        self._stack.close()
        self._rgb_queue = None
        self._detection_queue = None
