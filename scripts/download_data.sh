#!/bin/bash
# Download real educational datasets for replay evaluation.
# Usage: bash scripts/download_data.sh [duolingo|assistments|all]
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

download_duolingo() {
    echo "=== Downloading Duolingo Spaced Repetition Dataset (Settles & Meeder 2016) ==="
    DUOLINGO_DIR="$DATA_DIR/duolingo"
    mkdir -p "$DUOLINGO_DIR"

    if [ -f "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv" ]; then
        echo "Duolingo data already exists, skipping download."
        return
    fi

    # Harvard Dataverse direct download (file ID: 3091087)
    echo "Downloading from Harvard Dataverse (361 MB compressed)..."
    wget -O "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv.gz" \
        "https://dataverse.harvard.edu/api/access/datafile/3091087" \
        --no-check-certificate 2>&1 || \
    curl -L -o "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv.gz" \
        "https://dataverse.harvard.edu/api/access/datafile/3091087"

    # Verify it's actually gzip
    if file "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv.gz" | grep -q "gzip"; then
        echo "Decompressing..."
        gunzip "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv.gz"
    else
        echo "WARNING: Downloaded file is not gzip. Checking if it's already CSV..."
        mv "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv.gz" \
           "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv"
    fi

    echo "Duolingo download complete."
    ls -lh "$DUOLINGO_DIR/"
}

download_assistments() {
    echo "=== Downloading ASSISTments 2012-2013 Dataset ==="
    ASSIST_DIR="$DATA_DIR/assistments"
    mkdir -p "$ASSIST_DIR"

    if [ -f "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv" ]; then
        echo "ASSISTments data already exists, skipping download."
        return
    fi

    # Google Drive download (file ID: 1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY)
    GDRIVE_ID="1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY"
    echo "Downloading ASSISTments 2012-2013 from Google Drive..."

    # gdown handles Google Drive large-file confirmation prompts
    if command -v gdown &> /dev/null; then
        gdown "$GDRIVE_ID" -O "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv"
    else
        echo "Installing gdown for Google Drive download..."
        pip install gdown --quiet
        gdown "$GDRIVE_ID" -O "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv"
    fi

    # If downloaded as zip, unzip it
    if file "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv" | grep -q "Zip\|zip"; then
        echo "File is zipped, extracting..."
        mv "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv" \
           "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv.zip"
        cd "$ASSIST_DIR" && unzip -o "2012-2013-data-with-predictions-4-final.csv.zip"
        cd - > /dev/null
    fi

    echo "ASSISTments download complete."
    ls -lh "$ASSIST_DIR/"
}

# Parse argument
DATASET=${1:-all}

case $DATASET in
    duolingo)
        download_duolingo
        ;;
    assistments)
        download_assistments
        ;;
    all)
        download_duolingo
        download_assistments
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [duolingo|assistments|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
echo "Data directory contents:"
du -sh "$DATA_DIR"/*/ 2>/dev/null || echo "(empty)"
