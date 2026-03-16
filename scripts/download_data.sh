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

    # Harvard Dataverse direct download
    echo "Downloading from Harvard Dataverse (361 MB compressed)..."
    wget -O "$DUOLINGO_DIR/learning_traces.13m.csv.gz" \
        "https://dataverse.harvard.edu/api/access/datafile/5597065" \
        --no-check-certificate 2>/dev/null || \
    curl -L -o "$DUOLINGO_DIR/learning_traces.13m.csv.gz" \
        "https://dataverse.harvard.edu/api/access/datafile/5597065"

    echo "Decompressing..."
    gunzip "$DUOLINGO_DIR/learning_traces.13m.csv.gz" 2>/dev/null || \
    gzip -d "$DUOLINGO_DIR/learning_traces.13m.csv.gz"
    mv "$DUOLINGO_DIR/learning_traces.13m.csv" \
       "$DUOLINGO_DIR/settles.acl16.learning_traces.13m.csv" 2>/dev/null || true

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

    # Direct download from ASSISTments data site
    echo "Downloading ASSISTments 2012-2013 dataset..."
    wget -O "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv.zip" \
        "https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect/2012-2013-data-with-predictions-4-final.csv.zip?attredirects=0&d=1" \
        --no-check-certificate 2>/dev/null || \
    curl -L -o "$ASSIST_DIR/2012-2013-data-with-predictions-4-final.csv.zip" \
        "https://drive.google.com/uc?export=download&id=0B5sVDqq4sGXUY2VYZmZyb3lFckk"

    echo "Unzipping..."
    cd "$ASSIST_DIR" && unzip -o "2012-2013-data-with-predictions-4-final.csv.zip" 2>/dev/null || true
    cd - > /dev/null

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
