#!/bin/bash
sudo kill $(pgrep -f "gst-launch-1.0 udpsrc port=")
sudo modprobe -r v4l2loopback
echo "pipelines stopped"
