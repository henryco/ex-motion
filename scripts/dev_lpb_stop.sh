#!/bin/bash
kill $(pgrep -f "gst-launch-1.0 udpsrc port=")
kill $(pgrep -f "dev_lpb_start.sh")
sudo modprobe -r v4l2loopback
echo "pipelines stopped"
