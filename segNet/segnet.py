#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

import jetson.inference
import jetson.utils

from jetson.inference import actionNet
from jetson.utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify the action/activity of an image sequence.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=actionNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="display://0", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet-18", help="pre-trained model to load (see below for options)")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)


output = videoOutput(args.output)


# load the recognition network
net = actionNet("resnet-18")


# create video sources & outputs
input = videoSource(args.input, argv=['--input-codec=H265'])
font = cudaFont()

# process frames until EOS or the user exits
while output.IsStreaming():
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
    output.Render(img)

    # update the title bar
    output.SetStatus("actionNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()