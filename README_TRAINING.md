# Training Your Own Rock-Paper-Scissors Model

This guide will help you train a highly accurate gesture recognition model using your own hand data.

## Step 1: Collect Training Data

You have two options for collecting training data:

### Option A: Download Images from the Web (Recommended for Quick Start)

Download hand gesture images automatically from the web:

```bash
python3 download_training_images.py
```

**What it does:**
- Searches the web for rock, paper, and scissors hand gesture images
- Downloads 200 images per class automatically
- Uses specific search queries to find relevant hand gestures
- Validates and filters images for quality

**Benefits:**
- ✅ Fast - no manual collection needed
- ✅ Diverse images from different sources
- ✅ Good starting point for training

**Note:** The script may take 10-20 minutes due to rate limiting. It handles delays automatically.

### Option B: Collect Your Own Images (Better for Personalization)

Run the data collection script with your webcam:

```bash
python3 collect_data.py
```

**Instructions:**
1. The script will ask you to choose a class (rock/paper/scissors)
2. Position your hand in the yellow box on screen
3. Press **SPACE** to capture an image
4. Press **ESC** to finish collecting for that class
5. Repeat for all three classes

**Goal:** Collect at least 200-300 images per class
- Vary angles, distances, lighting
- Different hand positions
- Try with/without sleeves

### Option C: Automatic Collection

For continuous automatic collection:

```bash
python3 collect_data_auto.py
```

This automatically captures images every 0.3 seconds when you press SPACE.

## Step 2: Train the Model

Once you have collected data, train the model:

```bash
python3 train_model.py
```

This will:
- Load your collected images
- Train a MobileNetV2 model (transfer learning)
- Save the model to `rps_model.pth`
- Save class names to `class_names.json`

**Training takes a few minutes** depending on your data size.

## Step 3: Use the Trained Model

The main app (`app.py`) will automatically detect and use your trained model if it exists!

**Benefits of using the trained model:**
- ✅ Much more accurate than OpenCV-based detection
- ✅ Works with YOUR specific hand characteristics
- ✅ Handles different lighting and angles better
- ✅ Shows confidence scores

## Tips for Better Accuracy

1. **More data = better accuracy**
   - Aim for 300-500 images per class
   - More variety = better generalization

2. **Balanced classes**
   - Same number of images for rock, paper, and scissors

3. **Good lighting**
   - Consistent lighting helps
   - But vary it slightly in training data

4. **Clear gestures**
   - Make distinct gestures
   - Rock: tight fist
   - Paper: fully open hand
   - Scissors: clear two-finger gesture

## File Structure

```
fcdtec/
├── data/
│   ├── rock/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── paper/
│   └── scissors/
├── rps_model.pth              # Trained model (created after training)
├── class_names.json            # Class names (created after training)
├── download_training_images.py # Download images from web (Option A)
├── collect_data.py             # Manual data collection script (Option B)
├── collect_data_auto.py        # Automatic data collection (Option C)
├── train_model.py              # Training script
└── app.py                      # Main app (uses model if available)
```

## Troubleshooting

**Model not loading?**
- Make sure `rps_model.pth` and `class_names.json` exist
- Check that PyTorch is installed: `pip install torch torchvision`

**Low accuracy?**
- Collect more training data
- Ensure images are clear and well-lit
- Make sure gestures are distinct

**Training errors?**
- Check that you have images in `data/rock/`, `data/paper/`, `data/scissors/`
- Ensure PyTorch is properly installed

