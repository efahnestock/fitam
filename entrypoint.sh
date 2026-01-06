#!/bin/bash
set -e

# Get target UID/GID from environment (passed by docker-compose)
TARGET_UID=${LOCAL_UID:-1000}
TARGET_GID=${LOCAL_GID:-1000}
USERNAME=developer

# If running as root, set up the user and drop privileges
if [ "$(id -u)" = "0" ]; then
    # Check if group exists with the target GID, create if not
    if ! getent group "$TARGET_GID" > /dev/null 2>&1; then
        groupadd -g "$TARGET_GID" "$USERNAME" 2>/dev/null || groupmod -g "$TARGET_GID" "$USERNAME"
    fi
    
    # Get the group name for this GID
    GROUP_NAME=$(getent group "$TARGET_GID" | cut -d: -f1)
    
    # Check if user exists, create or modify as needed
    if id "$USERNAME" > /dev/null 2>&1; then
        # User exists, modify UID/GID if different
        usermod -u "$TARGET_UID" -g "$TARGET_GID" "$USERNAME" 2>/dev/null || true
    else
        # Create user with target UID/GID
        useradd -u "$TARGET_UID" -g "$TARGET_GID" -m -s /bin/bash "$USERNAME"
    fi
    
    # Ensure home directory ownership is correct
    chown -R "$TARGET_UID:$TARGET_GID" /home/"$USERNAME" 2>/dev/null || true
    
    # Execute the command as the target user
    exec gosu "$USERNAME" "$@"
else
    # Not running as root, just execute the command
    exec "$@"
fi
