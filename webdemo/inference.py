# Copyright 2021 The TensorFlow Authors
# Copyright 2024 ISNing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run image classification."""

import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

mean = [127.5, 127.5, 127.5]
std = [127.5, 127.5, 127.5]


class Classifier:
    start_time = time.time()
    base_options = None
    classification_options = None
    options = None
    classifier = None

    def __init__(self, model: str, max_results: int, score_threshold: float, num_threads: int) -> None:
        """Continuously run inference on images acquired from the camera.

        Args:
          model: Name of the TFLite image classification model.
          max_results: Max of classification results.
          score_threshold: The score threshold of classification results.
          num_threads: Number of CPU threads to run the model.
        """

        # Initialize the image classification model
        self.base_options = core.BaseOptions(
            file_name=model, num_threads=num_threads)
        # Enable Coral by this setting
        self.classification_options = processor.ClassificationOptions(
            max_results=max_results, score_threshold=score_threshold)
        self.options = vision.ImageClassifierOptions(
            base_options=self.base_options, classification_options=self.classification_options)

        self.classifier = vision.ImageClassifier.create_from_options(self.options)

    def inference(self, image):
        # Create TensorImage from the RGB image
        tensor_image = vision.TensorImage.create_from_array(image)
        # List classification results
        categories = self.classifier.classify(tensor_image)

        return categories
