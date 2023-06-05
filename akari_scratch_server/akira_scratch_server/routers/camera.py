from akari_scratch_server.media import CaptureMode
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ._context import get_context
from ..captures.detection_type import GetDetectionResponse

router = APIRouter()


@router.get("/stream")
async def get_stream() -> StreamingResponse:
    context = get_context()
    return StreamingResponse(
        context.media_controller.consumer(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class SetCaptureModeRequest(BaseModel):
    mode: CaptureMode


class GetCaptureModeResponse(BaseModel):
    mode: CaptureMode


@router.post("/mode")
def set_mode(request: SetCaptureModeRequest) -> None:
    context = get_context()
    context.media_controller.switch_mode(request.mode)


@router.get("/mode", response_model=GetCaptureModeResponse)
def get_mode() -> GetCaptureModeResponse:
    context = get_context()
    return GetCaptureModeResponse(
        mode=context.media_controller.mode,
    )


@router.get("/face", response_model=GetDetectionResponse)
def get_face_result() -> GetDetectionResponse:
    context = get_context()
    result = GetDetectionResponse()
    if(context.media_controller != None):
        result = context.media_controller.get_face_result()
    return result

@router.get("/object", response_model=GetDetectionResponse)
def get_object_result() -> GetDetectionResponse:
    context = get_context()
    result = GetDetectionResponse()
    if(context.media_controller != None):
        result = context.media_controller.get_object_result()
    return result
