#!/bin/bash

# Ensure backend is running
# Usage: ./smoke_test.sh

API_URL="http://127.0.0.1:8000"
REPO_URL="https://github.com/miguelgrinberg/flasky.git"

echo "1. Indexing repository..."
INDEX_RES=$(curl -s -X POST "$API_URL/index" -H "Content-Type: application/json" -d "{\"repo_url\": \"$REPO_URL\"}")
JOB_ID=$(echo $INDEX_RES | grep -o '"job_id":"[^"]*' | grep -o '[^"]*$')

if [ -z "$JOB_ID" ]; then
    echo "Failed to start index job. Backend might be down."
    exit 1
fi

echo "Job ID: $JOB_ID"

echo "Waiting for indexing to complete..."
while true; do
    STATUS_RES=$(curl -s "$API_URL/index/status/$JOB_ID")
    STATUS=$(echo $STATUS_RES | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
    echo "Current status: $STATUS"
    if [ "$STATUS" == "done" ]; then
        break
    elif [[ "$STATUS" == error* ]]; then
        echo "Indexing failed!"
        exit 1
    fi
    sleep 5
done

echo "Indexing complete."

echo "2. Asking questions..."

QUESTIONS=(
    "How does the login function authenticate a user?"
    "What is the purpose of config.py in this project?"
    "How does this application connect to MongoDB?"
)

for Q in "${QUESTIONS[@]}"; do
    echo "----------------------------------------"
    echo "Q: $Q"
    ASK_RES=$(curl -s -X POST "$API_URL/ask" -H "Content-Type: application/json" -d "{\"question\": \"$Q\", \"mode\": \"proposed\"}")
    
    STATUS=$(echo $ASK_RES | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
    echo "STATUS: $STATUS"
    
    # Simple extraction for preview
    echo "Preview of response:"
    echo $ASK_RES | grep -o '"answer":"[^"]*' | head -c 150
    echo "..."
done
echo "----------------------------------------"
echo "Smoke test complete."
