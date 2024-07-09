import jetson.inference
import jetson.utils

# Initialize the display output
display = jetson.utils.videoOutput("display://0")

# Load the object detection network
net = jetson.inference.detectNet("dashcamnet", threshold=0.5)

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

    person_count = 0
    car_count = 0
    bike_count = 0
    sign_count = 0

    for d in detections:
        className = net.GetClassDesc(d.ClassID)
        if className == "person":
            person_count+=1

        elif className == "car":
            car_count+=1

        elif className == "bike":
            bike_count+=1

        elif className == "sign":
            sign_count+=1

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS | Person {:.0f} | Car {:.0f} | Bike {:.0f} | Sign {:.0f}".format(net.GetNetworkFPS(), person_count, car_count, bike_count, sign_count))