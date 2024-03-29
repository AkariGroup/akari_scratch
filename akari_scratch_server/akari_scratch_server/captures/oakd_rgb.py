import contextlib
from typing import Optional

import cv2
import depthai as dai
import numpy

OUTPUT_WIDTH =480
OUTPUT_HEIGHT =360

class RGBCapture:
    @staticmethod
    def _create_pipeline() -> dai.Pipeline:
        pipeline = dai.Pipeline()

        # camera setting
        rgb_camera = pipeline.create(dai.node.ColorCamera)
        rgb_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        
        # video setting
        video = pipeline.create(dai.node.XLinkOut)
        video.setStreamName("video")
        video.input.setBlocking(False)
        video.input.setQueueSize(1)
        rgb_camera.video.link(video.input)

        return pipeline

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()
        self._pipeline = RGBCapture._create_pipeline()
        # NOTE: usb2Mode make the x-device-link connection stable
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        self._video: Optional[dai.DataOutputQueue] = self._device.getOutputQueue(  # type: ignore
            name="video",
            maxSize=1,
            blocking=False,
        )

    def get_frame(self) -> Optional[numpy.ndarray]:
        video = self._video
        if video is None:
            return None
        frame: Optional[numpy.ndarray] = video.get().getCvFrame()  # type: ignore
        frame: np.ndarray = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        return frame

    def close(self) -> None:
        self._stack.close()
        self._video = None
