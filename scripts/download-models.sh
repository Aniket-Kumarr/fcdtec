#!/usr/bin/env bash

# Script to download face-api.js models
MODELS_DIR="public/models"
BASE_URL="https://raw.githubusercontent.com/justadudewhohacks/face-api.js-models/master"

mkdir -p "$MODELS_DIR"

echo "Downloading face-api.js models..."

# Tiny Face Detector
curl -L "$BASE_URL/tiny_face_detector_model-weights_manifest.json" -o "$MODELS_DIR/tiny_face_detector_model-weights_manifest.json"
curl -L "$BASE_URL/tiny_face_detector_model-shard1" -o "$MODELS_DIR/tiny_face_detector_model-shard1"

# Face Landmark 68
curl -L "$BASE_URL/face_landmark_68_model-weights_manifest.json" -o "$MODELS_DIR/face_landmark_68_model-weights_manifest.json"
curl -L "$BASE_URL/face_landmark_68_model-shard1" -o "$MODELS_DIR/face_landmark_68_model-shard1"

# Face Recognition
curl -L "$BASE_URL/face_recognition_model-weights_manifest.json" -o "$MODELS_DIR/face_recognition_model-weights_manifest.json"
curl -L "$BASE_URL/face_recognition_model-shard1" -o "$MODELS_DIR/face_recognition_model-shard1"
curl -L "$BASE_URL/face_recognition_model-shard2" -o "$MODELS_DIR/face_recognition_model-shard2"

# Face Expression
curl -L "$BASE_URL/face_expression_model-weights_manifest.json" -o "$MODELS_DIR/face_expression_model-weights_manifest.json"
curl -L "$BASE_URL/face_expression_model-shard1" -o "$MODELS_DIR/face_expression_model-shard1"

echo "Models downloaded successfully to $MODELS_DIR"


