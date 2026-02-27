import pytest
import numpy as np
import cv2
from src.core.cv_pipeline import CVPipeline

def test_cv_pipeline_initialization():
    pipeline = CVPipeline()
    assert pipeline is not None
    assert pipeline.hands is not None

def test_process_frame_no_hands():
    pipeline = CVPipeline()
    # Create a blank black frame (no hands)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    overlay_frame, hand_stats, hands_present = pipeline.process_frame(test_frame)
    
    assert overlay_frame is not None
    assert hand_stats is None
    assert hands_present is None
