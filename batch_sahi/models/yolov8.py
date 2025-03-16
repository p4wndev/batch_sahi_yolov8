from sahi.models.base import DetectionModel
from typing import List
import numpy as np

class Yolov8DetectionModelCustom(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from ultralytics import YOLO

        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)
        
    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping
            
    def perform_inference(self, images: List[np.ndarray]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        prediction_result = self.model.predict(source=images, verbose=False, device=self.device)

        prediction_result = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold] for result in prediction_result
        ]

        self._original_predictions = prediction_result
