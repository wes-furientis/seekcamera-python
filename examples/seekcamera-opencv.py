#!/usr/bin/env python3
# Copyright 2021 Seek Thermal Inc.
#
# Original author: Michael S. Mead <mmead@thermal.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import signal
import sys
from threading import Condition, Thread, Event

import cv2
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viewer.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("viewer")

from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCameraAGCMode,
    SeekCameraFilter,
    SeekCameraFilterState,
    SeekCameraFlatSceneCorrectionID,
    SeekCameraHistEQAGCGainLimitFactorMode,
    SeekCameraHistEQAGCPlateauRedistributionMode,
    SeekCameraPipelineMode,
    SeekCamera,
    SeekFrame,
)


class Renderer:
    """Contains camera and image data required to render images to the screen."""

    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True


def on_frame(_camera, camera_frame, renderer):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------
    _camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    renderer: Renderer
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the renderer object.
    """

    # Acquire the condition variable and notify the main thread
    # that a new frame is ready to render. This is required since
    # all rendering done by OpenCV needs to happen on the main thread.
    with renderer.frame_condition:
        renderer.frame = camera_frame.color_argb8888
        renderer.frame_count = getattr(renderer, 'frame_count', 0) + 1
        renderer.frame_condition.notify()


def on_event(camera, event_type, event_status, renderer):
    """Async callback fired whenever a camera event occurs.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera on which an event occurred.
    event_type: SeekCameraManagerEvent
        Enumerated type indicating the type of event that occurred.
    event_status: Optional[SeekCameraError]
        Optional exception type. It will be a non-None derived instance of
        SeekCameraError if the event_type is SeekCameraManagerEvent.ERROR.
    renderer: Renderer
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the Renderer object.
    """
    log.info("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        if renderer.busy:
            return

        # Claim the renderer.
        # This is required in case of multiple cameras.
        renderer.busy = True
        renderer.camera = camera

        # Indicate the first frame has not come in yet.
        # This is required to properly resize the rendering window.
        renderer.first_frame = True

        # Configure Eagle pipeline mode BEFORE starting capture.
        try:
            camera.pipeline_mode = SeekCameraPipelineMode.EAGLE
            log.info("Pipeline mode set to EAGLE")
        except Exception as e:
            log.info("Failed to set EAGLE pipeline mode: {}".format(e))

        # Set color palette before capture.
        camera.color_palette = SeekCameraColorPalette.TYRIAN

        # Start imaging.
        camera.register_frame_available_callback(on_frame, renderer)
        camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)
        log.info("Capture session started")

        # AGC must be set AFTER capture session starts.
        camera.agc_mode = SeekCameraAGCMode.HISTEQ
        log.info("AGC mode set to HISTEQ")

        # Tune HistEQ AGC (modeled after FLIR Boson defaults).
        # Plateau: limits max % of pixels per histogram bin. Prevents noise
        # amplification on flat scenes. FLIR defaults to 7%.
        camera.histeq_agc_plateau = 0.07
        # Redistribute clipped pixels among active bins for better contrast.
        camera.histeq_agc_plateau_redistribution_mode = (
            SeekCameraHistEQAGCPlateauRedistributionMode.ACTIVE_BINS_ONLY
        )
        # Gain limit: caps max contrast gain. FLIR defaults to 1.38.
        camera.histeq_agc_gain_limit = 1.5
        camera.histeq_agc_gain_limit_factor_mode = (
            SeekCameraHistEQAGCGainLimitFactorMode.AUTO
        )
        # Trim histogram tails to exclude outlier pixels.
        camera.histeq_agc_trim_left = 0.005
        camera.histeq_agc_trim_right = 0.005
        # Temporal smoothing to reduce frame-to-frame flicker.
        camera.histeq_agc_alpha_time = 1.0
        log.info("HistEQ AGC tuned: plateau=0.07, gain_limit=1.5, trim=0.5%")

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        # Check that the camera disconnecting is one actually associated with
        # the renderer. This is required in case of multiple cameras.
        if renderer.camera == camera:
            # Stop imaging and reset all the renderer state.
            camera.capture_session_stop()
            renderer.camera = None
            renderer.frame = None
            renderer.busy = False

    elif event_type == SeekCameraManagerEvent.ERROR:
        log.info("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return


def main():
    window_name = "Seek Thermal - Python OpenCV Sample"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    keep_running = True
    shutdown_event = Event()

    def signal_handler(sig, frame):
        nonlocal keep_running
        if not keep_running:
            # Second Ctrl+C — force exit immediately
            log.info("Force exit.")
            os._exit(1)
        log.info("Shutting down... (Ctrl+C again to force)")
        keep_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    manager = SeekCameraManager(SeekCameraIOType.USB)
    renderer = Renderer()
    manager.register_event_callback(on_event, renderer)

    while keep_running:
        with renderer.frame_condition:
            if renderer.frame_condition.wait(150.0 / 1000.0):
                img = renderer.frame.data

                if renderer.first_frame:
                    (height, width, _) = img.shape
                    cv2.resizeWindow(window_name, width * 2, height * 2)
                    renderer.first_frame = False

                cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("f") and renderer.camera is not None:
            log.info("Storing flat scene correction (FSC)...")
            try:
                renderer.camera.store_flat_scene_correction(
                    SeekCameraFlatSceneCorrectionID.ID_0
                )
                log.info("FSC stored successfully.")
            except Exception as e:
                log.info("FSC store failed: {}".format(e))
        elif key == ord("d") and renderer.camera is not None:
            log.info("Deleting flat scene correction...")
            try:
                renderer.camera.capture_session_stop()
                renderer.camera.delete_flat_scene_correction(
                    SeekCameraFlatSceneCorrectionID.ID_0
                )
                log.info("FSC deleted.")
                renderer.camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)
            except Exception as e:
                log.info("FSC delete failed: {}".format(e))

        try:
            if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                break
        except cv2.error:
            break

    # Attempt clean shutdown with a timeout — force exit if it hangs.
    cv2.destroyAllWindows()
    log.info("Cleaning up camera...")

    def cleanup():
        try:
            if renderer.camera is not None and renderer.busy:
                renderer.camera.capture_session_stop()
        except Exception:
            pass
        try:
            manager.destroy()
        except Exception:
            pass
        shutdown_event.set()

    cleanup_thread = Thread(target=cleanup, daemon=True)
    cleanup_thread.start()
    cleanup_thread.join(timeout=10)

    if cleanup_thread.is_alive():
        log.info("Cleanup timed out, forcing exit.")
        os._exit(0)

    log.info("Clean shutdown complete.")


if __name__ == "__main__":
    main()
