import jetson.inference
import jetson.utils

# Initialize the display output
display = jetson.utils.videoOutput("display://0")

# Load the object detection network
net = jetson.inference.detectNet("dashcamnet", threshold=0.3)

# Create the video source with the RTSP stream
rtsp_url = "rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
camera = jetson.utils.videoSource(rtsp_url, argv=['--input-codec=H265'])

# Process the video stream
while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        print("Failed to capture image from camera")
        continue

    # Ensure the image is in CUDA memory
    if not isinstance(img, jetson.utils.cudaImage):
        img = jetson.utils.cudaFromNumpy(img)

    detections = net.Detect(img, overlay="box,labels,conf")
    if detections is None:
        print("Failed to detect objects in the image")
        continue

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))