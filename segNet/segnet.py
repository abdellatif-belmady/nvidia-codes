import jetson.inference
from jetson.inference import segNet
import jetson.utils
from jetson.utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log
from segnet.utils import *



# load the segmentation network
net = segNet("fcn-resnet18-voc")

# set the alpha blending value
net.SetOverlayAlpha(150.0)

# create video output
output = videoOutput("display://0")

# create buffer manager
buffers = segmentationBuffers(net)

# create video source
rtsp_url = "rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
input = videoSource(rtsp_url, argv=['--input-codec=H265'])

# process frames until EOS or the user exits
while True:
    # capture the next image
    img_input = input.Capture()

    if img_input is None: # timeout
        continue
        
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)

    # process the segmentation network
    net.Process(img_input, ignore_class="void")

    # generate the overlay
    if buffers.overlay:
        net.Overlay(buffers.overlay, filter_mode="linear")

    # generate the mask
    if buffers.mask:
        net.Mask(buffers.mask, filter_mode="linear")

    # composite the images
    if buffers.composite:
        cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
        cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

    # render the output image
    output.Render(buffers.output)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("fcn-resnet18-voc", net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break