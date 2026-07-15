#!/bin/bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

# Define variables
USERNAME=huaj
LOCAL_DIR="/scratch"        # Path to the local directory
LOCAL_DIR2="/proj/"        # Path to the local directory
CONTAINER_DIR="/scratch"                # Path inside the container
CONTAINER_DIR2="/proj"                # Path inside the container
IMAGE="ullvm:0909"                       # Docker image to use
#COMMAND="/bin/bash -c \"cd /scratch/staff/huaj/hjacompiler/acompiler/\""                  # Command to run inside the container
COMMAND="/bin/bash"                  # Command to run inside the container
DOCKER_OPTS="-it --rm"               # Additional Docker options

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Local directory $LOCAL_DIR does not exist."
    exit 1
fi

# Run Docker with volume mapping
#docker run $DOCKER_OPTS -u $USERNAME -v "$LOCAL_DIR:$CONTAINER_DIR" -v "$LOCAL_DIR2:$CONTAINER_DIR2" "$IMAGE" "$COMMAND"
docker run $DOCKER_OPTS -v "$LOCAL_DIR:$CONTAINER_DIR" -v "$LOCAL_DIR2:$CONTAINER_DIR2" "$IMAGE" "$COMMAND"
#docker run $DOCKER_OPTS -v "$LOCAL_DIR:$CONTAINER_DIR" "$IMAGE" "$COMMAND"
