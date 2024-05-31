#!/bin/bash

# EXAMPLE INPUT OF A FILE:
# [ID=PORT]
# 200=5000
# 201=5001
#

filename=$1

if [[ ! -e "$filename" ]]; then
    echo "Error: File $filename does not exist."
    exit 1
fi

id_list=""
declare -A id_port_map

while IFS="=" read -r ID PORT || [[ -n "$ID" ]] ; do
  # trim strings
  ID=$(echo "$ID" | xargs)
  PORT=$(echo "$PORT" | xargs)

  if [ -z "$id_list" ]; then
      id_list="$ID"
  else
      id_list="$id_list,$ID"
  fi

  id_port_map["$ID"]="$PORT"
done < "$filename"

sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr="$id_list"

for id in "${!id_port_map[@]}"; do
  port="${id_port_map[$id]}"
  nohup gst-launch-1.0 udpsrc port="$port" caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=26, framerate=60/1" ! rtpjpegdepay ! jpegdec ! videoconvert ! v4l2sink device="/dev/video$id" &
done

echo "pipelines started"
