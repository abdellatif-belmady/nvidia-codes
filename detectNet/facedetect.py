import jetson.inference
import jetson.utils

# Initialize the display output
display = jetson.utils.videoOutput("display://0")

# Load the object detection network
net = jetson.inference.detectNet("facedetect", threshold=0.5)

# Create the video source with the RTSP stream
rtsp_url = "rtsp://admin:admin123@doorzddns.ddns.net:5006/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
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

    face_count = 0

    for d in detections:
        className = net.GetClassDesc(d.ClassID)
        if className == "face":
            face_count+=1

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS | Face {:.0f}".format(net.GetNetworkFPS(), face_count))