import jetson.inference
from jetson.inference import segNet
import jetson.utils
from jetson.utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log, cudaAllocMapped

class segmentationBuffers:
    def __init__(self, net):
        self.net = net
        self.overlay = None
        self.mask = None
        self.composite = None
        self.output = None

    def Alloc(self, shape, format):
        self.overlay = cudaAllocMapped(width=shape[1], height=shape[0], format=format)
        self.mask = cudaAllocMapped(width=shape[1], height=shape[0], format=format)
        self.composite = cudaAllocMapped(width=shape[1] * 2, height=shape[0], format=format)
        self.output = cudaAllocMapped(width=shape[1] * 2, height=shape[0], format=format)

# load the segmentation network
net = segNet("fcn-resnet18-voc")
print(f"Loaded segmentation network: fcn-resnet18-voc")

# set the alpha blending value
net.SetOverlayAlpha(150.0)

# create video output
output = videoOutput("display://0")
print("Created video output")

# create buffer manager
buffers = segmentationBuffers(net)
print("Created segmentation buffers")

# create video source
rtsp_url = "rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
input = videoSource(rtsp_url, argv=['--input-codec=H265'])
print(f"Created video source from: {rtsp_url}")

# process frames until EOS or the user exits
while True:
    # capture the next image
    img_input = input.Capture()
    
    if img_input is None:
        print("Failed to capture image")
        continue
    
    print(f"Captured image: {img_input.width}x{img_input.height}, format: {img_input.format}")
        
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)
    print("Allocated buffers")

    # process the segmentation network
    net.Process(img_input, ignore_class="void")
    print(f"Number of classes detected: {net.GetNumClasses()}")

    # generate the overlay
    if buffers.overlay:
        net.Overlay(buffers.overlay, filter_mode="linear")
        print("Overlay generated")

    # generate the mask
    if buffers.mask:
        net.Mask(buffers.mask, filter_mode="linear")
        print("Mask generated")

    # composite the images
    if buffers.composite:
        cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
        cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)
        print(f"Composite image created: {buffers.composite.width}x{buffers.composite.height}")

    # render the output image
    print(f"Rendering output: {buffers.output.width}x{buffers.output.height}")
    output.Render(buffers.output)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("fcn-resnet18-voc", net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        print("Stream ended")
        break

print("Script completed")