#!/bin/bash

# Define the source and destination directories
PREFIX="../src/"
FRIDOM_DIR="fridom"
SOURCE_DIR="$PREFIX$FRIDOM_DIR"
DEST_DIR="source/fridom"

# Function to create the directory structure
create_directories() {
    local SRC=$1
    local DST=$2

    # Create the destination directory if it doesn't exist
    mkdir -p "$DST"

    # Iterate over the items in the source directory
    for ITEM in "$SRC"/*; 
    do
        # Get the base name of the item
        BASENAME=$(basename "$ITEM")

        # Skip __pycache__ directories
        if [[ "$BASENAME" == "__pycache__" ]]; then
            continue
        fi

        # If the item is a directory, recursively call this function
        if [ -d "$ITEM" ]; 
        then
            create_directories "$ITEM" "$DST/$BASENAME"
        fi
    done
}

# Function to replicate the directory structure and rename files
replicate_structure() {
    local SRC=$1
    local DST=$2

    # Create the destination directory if it doesn't exist
    mkdir -p "$DST"

    # Initialize lists to keep track of subdirectories and files
    local subdirs=()
    local files=()

    # Iterate over the items in the source directory
    for ITEM in "$SRC"/*; 
    do
        # Get the base name of the item
        BASENAME=$(basename "$ITEM")

        # Skip __pycache__ directories
        if [[ "$BASENAME" == "__pycache__" ]]; then
            continue
        fi

        # If the item is a directory, recursively call this function
        if [ -d "$ITEM" ]; 
        then
            subdirs+=("$BASENAME")
            replicate_structure "$ITEM" "$DST/$BASENAME"
        else
            # Handle files
            if [[ "$BASENAME" == "__init__.py" ]]; 
            then
                # Replace __init__.py with index.rst
                if [[ ! -f "$DST/index.rst" ]]; then
                    create_index_rst "$DST" "$SRC" "$BASENAME"
                fi
            else
                files+=("${BASENAME%.py}")
                # Replace .py with .rst
                NEW_BASENAME="${BASENAME%.py}.rst"
                if [[ ! -f "$DST/$NEW_BASENAME" ]]; then
                    create_file_rst "$DST" "$SRC" "$BASENAME"
                fi
            fi
        fi
    done
}

# Function to create an index.rst file with the specified content
create_index_rst() {
    local DST=$1
    local SRC=$2
    local INIT_FILE=$3

    FOLDER_NAME=$(basename "$DST")
    RELATIVE_PATH="${SRC#$PREFIX}"

    # get local subdirectories and files
    local subdirs=()
    local files=()
    for ITEM in "$SRC"/*;
    do
        BASENAME=$(basename "$ITEM")
        if [[ "$BASENAME" == "__pycache__" ]]; then
            continue
        fi
        if [ -d "$ITEM" ]; then
            subdirs+=("$BASENAME")
        else
            if [[ "$BASENAME" != "__init__.py" ]]; then
                files+=("${BASENAME%.py}")
            fi
        fi
    done

    cat << EOF > "$DST/index.rst"
$FOLDER_NAME
=============

.. automodule:: ${RELATIVE_PATH//\//.}
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 1
   :caption: Modules:

EOF

    # Append subdirectories to the Modules section
    for subdir in "${subdirs[@]}"; do
        echo "   $subdir/index" >> "$DST/index.rst"
    done

    cat << EOF >> "$DST/index.rst"

.. toctree::
   :maxdepth: 1
   :caption: Classes:

EOF

    # Append files to the Classes section
    for file in "${files[@]}"; do
        echo "   $file" >> "$DST/index.rst"
    done
}

# Function to create a .rst file for a .py file with the specified content
create_file_rst() {
    local DST=$1
    local SRC=$2
    local BASENAME=$3

    FILE_NAME="${BASENAME%.py}"
    RELATIVE_PATH="${SRC#$PREFIX}"

    cat << EOF > "$DST/$FILE_NAME.rst"
$FILE_NAME
===========

.. autoclass:: ${RELATIVE_PATH//\//.}.$FILE_NAME
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
EOF
}

# First, create all directories
create_directories "$SOURCE_DIR" "$DEST_DIR"

# Call the function with the source and destination directories
replicate_structure "$SOURCE_DIR" "$DEST_DIR"
