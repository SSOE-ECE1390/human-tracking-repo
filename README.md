<!-- # People Counter & Tracker (with ROI)

## Team
- Shiqi Wang — shw322@pitt.edu
- Yuqi Xiong — YUX115@pitt.edu

## What this project does
- **Provide a Data Driven View of a soccer match**.
- **Count Players in Vedio** and **Find position of players** in a video.
- **Identify Players acording their Teams**  and **saves the ROI** (cropped area).

<video src="demo.mp4" controls width="600"> </video>

## Milestones / TODOs
- Project setup: repo structure and environment. -  by Sep 17
- Person detection for each frame (get counts). - by Oct 1
- Simple selection UI (click a box to choose the person). - by Oct 15
- Tracking the selected person (keep a stable ID). - by Oct 15 
- ROI extraction (save crops/video clips of the target). - by Nov 20
- Basic tests and example demo video. - by Nov 30
- Documentation updates as features land.

## How to Use

- Find a soccer match vedio: The camera should be relatively stable. Tactical Camera is ideal.
The model is trained by scource: "https://www.youtube.com/watch?v=h9C0DUJyE3M&t=5101s" 
- Once you download the soccer match video using python script in auto_generate_dataset to capture the frames for training or verification set
1 use create_frames.py to create pictures
2 use copyfiles_to_dataset.py to divid pictures into training and verification set. Normally training set should have larger number of pics

- Label data by hand is boring why not do it automated? We can use yolo11 Large to find person and draw the box first.
1 Using generate_training.py --generate to generate label which can identify humans in the frames.
2 Then using cleandata.py to git rid of weird labels
3 Once we have the humans ROIs we can use k-mean cluster on the mean RGB 3D space of ROIs to group and label the player by their teams. Using generate_training.py --kmeans to train kmeans
kmeans_1.png is our dataset's kmean RGB 3D dimension representation

4 Then we can use generate_training.py --generate again to apply kmeans to generate labels that acoarding to their teams or other personnel. 
5 We can use draw_box.py to see how the players are labeled. And see_player_indvidually.py can help us see all ROIs of one frame specifically.

- Train model
1 In our case be case the classes are relatively small, thus we dont need too advanced model I used yolo11n as the training target. Besides, it will be cooller if our trained simple model can out perform the original advanced model.
2 Using train_model.ipynb to train your model.

- Test model
1 prepare another vedio players wear the same shirt. 
2 Using track_real.py to see how your model perform! 
There are model train by kmeans as well as brightness throushold provided -->

# Soccer player Counter & Tracker (with ROI)

## Team

- Shiqi Wang — <shw322@pitt.edu>  
- Yuqi Xiong — <YUX115@pitt.edu>

---

## Overview

This project provides a **data-driven view of a soccer match** using computer vision and deep learning. It can:

- **Detect and count players** in each frame of a soccer video  
- **Track player positions over time**  
- **Identify players by team** using color-based clustering on their ROIs (cropped images)  
- **Extract and save ROIs** for further analysis or visualization  

The system is designed for tactical / broadcast-style soccer videos where the camera is relatively stable (tactical camera is ideal).

---

## Demo

If your browser and GitHub allow inline playback, you should see a demo here:

<video src="demo.mp4" controls width="600"></video>

If the embedded player does not show up, you can still open the demo video directly:

[▶️ Demo Video (demo.mp4)](demo.mp4)
[](demo.jpg)

---

## Features

- ✅ **Player detection** using a pretrained YOLO model (YOLOv11-large for automatic label generation, YOLOv11n for lightweight training)  
- ✅ **Automatic annotation pipeline** for generating human labels and cleaning noisy detections  
- ✅ **Color-based team clustering** via **k-means** in 3D RGB space of player ROIs  
- ✅ **Custom dataset generation** from YouTube / downloaded soccer matches  
- ✅ **Player tracking** over time with stable IDs  
- ✅ **ROI extraction** and saving cropped images / clips for each player or team  

---

## Data Source

The model in this repo is trained using frames extracted from the following source video:

> `"https://www.youtube.com/watch?v=h9C0DUJyE3M&t=5101s"`

You can replace this with your own soccer match video, as long as the camera is reasonably stable and players are visible.

---

## Project Structure (typical)

> The exact layout may vary slightly, but the main components are:

- `auto_generate_dataset/`  
  - `create_frames.py` — extract frames from raw match videos  
  - `copyfiles_to_dataset.py` — split frames into training / validation sets
- `auto_labeling_data/`
  - `generate_training.py` — generate labels using YOLO, and apply k-means to assign teams  
  - `cleandata.py` — clean noisy / incorrect labels  
  - `draw_box.py` — visualize bounding boxes and team labels on frames  
  - `see_player_individually.py` — visualize all ROIs (crops) for a given frame  
- `train_model.ipynb` — notebook to train a YOLOv11n model on the generated dataset  
- `track_real.py` — run the trained model on a new soccer video and track players in real time  
- `demo.mp4` — example demo video (output of the pipeline)  
- `README.md` — this file  

---

## Setup

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>.git
   cd project


2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   Make sure you have at least:

   * Python 3.9+
   * OpenCV (`opencv-python`)
   * PyTorch
   * Ultralytics YOLO (v8/v11)
   * NumPy, scikit-learn (for k-means), etc.

   Example:

   ```bash
   pip install opencv-python torch torchvision torchaudio ultralytics scikit-learn numpy matplotlib
   ```

   (Adjust according to your environment and CUDA setup.)

---

## How to Use

### 1. Collect / Download a Soccer Match Video

* Choose a match video where:

  * The camera is **relatively stable**
  * The **entire field** and players are clearly visible
* Download it (e.g., from YouTube) into the project directory.
* The original model was trained from the video:
  `https://www.youtube.com/watch?v=h9C0DUJyE3M&t=5101s`

---

### 2. Generate Frames for Training / Validation

In `auto_generate_dataset/`:

1. **Extract frames**

   ```bash
   python3 create_frames.py --input your_match_video.mp4 --output frames/
   ```

   This script will save frames from the video into `frames/`.

2. **Split frames into train / validation**

   ```bash
   python3 copyfiles_to_dataset.py --input frames/ --train train_images/ --val val_images/
   ```

   * Training set should usually contain **more** images than the validation set.

---

### 3. Automatic Labeling & Data Cleaning

We first use a strong YOLO model (e.g., YOLOv11-large) to generate **person bounding boxes**, then clean and cluster them by team.

1. **Generate person labels**

   ```bash
   python3 generate_training.py --generate
   ```

   * Uses YOLOv11-large to detect persons in each frame
   * Saves bounding boxes and initial labels to annotation files

2. **Clean noisy labels**

   ```bash
   python3 cleandata.py
   ```

   * Removes weird / obviously incorrect detections
   * Ensures a cleaner training set

3. **K-means clustering on ROIs to separate teams**

   Player ROIs (cropped bounding boxes) are embedded into 3D RGB space (mean color, etc.), then clustered:

   ```bash
   python3 generate_training.py --kmeans
   ```

   * Runs k-means clustering on player ROIs
   * Assigns each player to a team (or other group) based on color
   * `kmeans_1.png` visualizes the distribution of our dataset in RGB space

   After k-means is trained:

4. **Generate final team-aware labels**

   ```bash
   python3 generate_training.py --generate --use-kmeans
   ```

   * Applies k-means results to label each detected player by team
   * Produces labels suitable for training the final YOLO model

5. **Visualize labels**

   * `draw_box.py` — draw bounding boxes with team labels on frames
   * `see_player_individually.py` — display all ROIs for a specific frame for inspection

---

### 4. Train the Detection Model

We use YOLOv11n (a small, lightweight model) as the final training target.

1. Open the notebook:

   ```bash
   jupyter notebook train_model.ipynb
   ```

2. In `train_model.ipynb`:

   * Configure the dataset paths (train / val images & labels)
   * Adjust hyperparameters (epochs, batch size, image size, etc.)
   * Start training

The idea is:

* YOLOv11-large is used only as a **teacher / label generator**
* YOLOv11n is a **smaller student model** that we train from scratch or finetune
* It is interesting to see if our lightweight model can **match or even outperform** the original large model on this specific task.

---

### 5. Run Tracking on a New Match

Once you have a trained model:

1. Prepare another soccer video where players wear **similar jerseys** (so k-means / color features still generalize).

2. Run:

   ```bash
   python3 track_real.py --video new_match.mp4 --weights path/to/your_trained_weights.pt
   ```

   This will:

   * Detect players frame by frame
   * Track them with consistent IDs
   * Draw bounding boxes and labels by team
   * Optionally save ROIs / cropped clips per player

There are also variants of the model trained with:

* k-means color clustering
* brightness thresholding

You can switch between them by changing the weights file and configuration inside `track_real.py`.

---

## How It Works (Pipeline Summary)

1. **Data Extraction**

   * Extract frames from raw soccer videos
   * Split into train / val sets

2. **Automatic Human Detection & ROI Extraction**

   * Use YOLOv11-large to detect person bounding boxes
   * Crop and save ROIs for each player

3. **Color-based Team Clustering (k-means)**

   * Represent each ROI in 3D RGB space (mean color)
   * Run k-means to cluster players into teams
   * Visualize the result (`kmeans_1.png`)

4. **Final Label Generation**

   * Apply k-means labels to detections
   * Generate training labels for “team-aware” player detection

5. **Training a Lightweight YOLO Model**

   * Train YOLOv11n on the generated dataset
   * Evaluate performance on validation frames

6. **Tracking & ROI Saving**

   * Run the trained model on new videos
   * Track players, maintain consistent IDs
   * Save per-player ROIs / clips for further tactical analysis

---

## Milestones / TODOs

* ✅ Project setup: repo structure and environment (by Sep 17)
* ✅ Person detection for each frame (get counts) (by Oct 1)
* ✅ Simple selection UI (click a box to choose a person) (by Oct 15)
* ✅ Tracking the selected person (stable ID) (by Oct 15)
* ✅ ROI extraction (save crops / clips of the target) (by Nov 20)
* ✅ Basic tests and example demo video (by Nov 30)
* ⬜ Further documentation and UI polish
* ⬜ Improved tracking robustness under occlusion / camera movement
* ⬜ Support for multiple leagues / different jersey styles

---

## Acknowledgements

* Original match video and data source from YouTube (linked above).
* YOLO-based models provided via the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.
* This project was developed as part of coursework at the University of Pittsburgh.

