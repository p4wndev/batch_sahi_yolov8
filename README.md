## Usage

```
from sahi_batch.models import Yolov8DetectionModelCustom
from sahi_batch import get_sliced_prediction_batch
```

```
Yolov8DetectionModelCustom(
    model_path="./yolov8n.pt",
    confidence_threshold=0.3,
    device="cuda:0",
    ...
)


get_sliced_prediction_batch(...)
```
