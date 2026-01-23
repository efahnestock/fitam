#!/bin/bash

# Run docker compose with current user's UID/GID
# Use 'env' to avoid conflict with readonly UID variable in bash
#env UID=$(id -u) GID=$(id -g) docker compose build --no-cache fitam
env UID=$(id -u) GID=$(id -g) docker compose run --rm -v /mnt/flex-s-pypzpbfqm6:/mnt/flex-s-pypzpbfqm6 fitam "$@"
