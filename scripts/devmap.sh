#!/bin/bash

# EXAMPLE INPUT OF A FILE:
#
# usb-0000:02:00.0-5=/dev/video100
# usb-0000:02:00.0-8=/dev/video101
# usb-0000:08:00.3-1.1=/dev/video102
# usb-0000:08:00.3-3=/dev/video103
#

filename=$1

if [[ ! -e "$filename" ]]; then
    echo "Error: File $filename does not exist."
    exit 1
fi

while IFS="=" read -r BUS_ID DEV_ID || [[ -n "$BUS_ID" ]] ; do
	# trim strings
	BUS_ID=$(echo "$BUS_ID" | xargs)
	DEV_ID=$(echo "$DEV_ID" | xargs)

	# creating new symlinks
	CUR_ID=$(v4l2-ctl --list-devices | grep "$BUS_ID" -A 2 | sed -n '2p' | sed 's/^[ \t]*//;s/[ \t]*$//')
	sudo ln -s "$CUR_ID" "$DEV_ID"
	echo "Created symlink for [$BUS_ID]:$CUR_ID -> $DEV_ID"
done < "$filename"
