import jetson.inference
import jetson.utils


display = jetson.utils.videoOutput("display://0")


# load the recognition network
net = jetson.inference.actionNet("resnet-18")


# create video sources & displays
input = jetson.utils.videoSource("rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", argv=['--input-codec=H265'])
font = jetson.utils.cudaFont()

# process frames until EOS or the user exits
while display.IsStreaming():
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    if not isinstance(img, jetson.utils.cudaImage):
        img = jetson.utils.cudaFromNumpy(img)

    # classify the action sequence
    class_id, confidence = net.Classify(img)
    class_desc = net.GetClassDesc(class_id)
    
    print(f"actionnet:  {confidence * 100:2.5f}% class #{class_id} ({class_desc})")
    
    # overlay the result on the image	
    font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)

    # render the image
    display.Render(img)

    # update the title bar
    display.SetStatus("actionNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()