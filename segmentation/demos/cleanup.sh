#!/bin/bash

# Find and delete directories containing "TEST" in their names
find . -type d -name '*TEST*' -exec rm -r {} +

echo "All folders containing 'TEST' have been deleted."

