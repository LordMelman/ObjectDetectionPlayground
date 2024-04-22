import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 

model_path = '/absolute/path/to/lite-model_efficientdet_lite0_detection_metadata_1.tflite'

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.detections.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

# define a video capture object 
vid = cv2.VideoCapture(0) 

with ObjectDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
