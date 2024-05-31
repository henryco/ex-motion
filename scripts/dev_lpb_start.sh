#!/bin/bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=200,201

nohup gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=26" ! rtpjpegdepay ! jpegdec ! videoconvert ! v4l2sink device=/dev/video200 &
nohup gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=26" ! rtpjpegdepay ! jpegdec ! videoconvert ! v4l2sink device=/dev/video201 &

echo "pipelines started"
