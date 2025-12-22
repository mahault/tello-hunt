# Tello Hunt

Autonomous drone tracking experiments using DJI Tello/Tello Talent with YOLO object detection.

## Features

- **Person Tracking**: Autonomous search, approach, signal, and backoff behavior
- **Cat Shadowing**: Follow cats while avoiding people (safety-first design)
- **Failsafe Landing**: All scripts include emergency landing on quit or error

## Requirements

- Python 3.10+
- DJI Tello or Tello Talent drone
- YOLOv8n weights file

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate tello-hunt

# Or with pip
pip install opencv-python djitellopy ultralytics keyboard

# Download YOLO weights (run once)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Getting Started

### Step 1: Connect to Drone

1. Power on the drone
2. Connect your computer to the drone's WiFi (TELLO-XXXXXX)

### Step 2: Test Command Connection

```bash
python cmd_debug.py
```

This tests the SDK command interface:
- Connects to drone and shows battery level
- Non-blocking keyboard controls: T=takeoff, L=land, Q=quit, E=emergency
- Prints height telemetry every 2 seconds while running

Use this to verify your drone responds to commands before testing video.

### Step 3: Test Video Stream

```bash
python video_debug_pyav.py
```

This tests the video stream:
- Starts the H.264 video stream on UDP port 11111
- Opens a window displaying live video from the drone
- Press Q in the window to quit

Use this to verify video works before running tracking applications.

### Step 4: Run Applications

Once both command and video work:

```bash
# Person tracking demo
python person_hunter_safe.py

# Cat shadowing with person avoidance
python cat_safe_shadow.py
```

## Scripts

### Debugging Tools

| Script | Purpose |
|--------|---------|
| `cmd_debug.py` | Test SDK commands - battery queries, takeoff/land with keyboard |
| `video_debug_pyav.py` | Test video stream - verify H.264 decoding works |

### Applications

| Script | Description |
|--------|-------------|
| `person_hunter_safe.py` | Search for people, approach, signal when in range, back off and land |
| `cat_safe_shadow.py` | Shadow cats at safe distance; backs away if person detected |

### Utilities

| Script | Description |
|--------|-------------|
| `fly_square.py` | Simple autonomous square flight pattern |
| `tello_connect_wifi.py` | One-time setup to connect drone to home WiFi |

## Controls

| Script | Start | Stop | Other |
|--------|-------|------|-------|
| `cmd_debug.py` | T | Q | L=land, E=emergency |
| `video_debug_pyav.py` | Auto | Q (window) | - |
| `person_hunter_safe.py` | ENTER | Q (window) | - |
| `cat_safe_shadow.py` | T | Q | H = hover |
| `fly_square.py` | Auto | Q (hold key) | - |

## Troubleshooting

### Drone not responding to commands

1. Verify WiFi connection to TELLO-XXXXXX
2. Run `cmd_debug.py` - should show battery percentage
3. Check battery (won't respond if critically low)
4. Default IP is `192.168.10.1` for direct connection mode

### Video stream not working

1. First confirm commands work with `cmd_debug.py`
2. Run `video_debug_pyav.py`
3. Check that port 11111 isn't blocked by firewall
4. Try power cycling the drone

### YOLO model missing

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Safety

- Fly in open spaces away from people and obstacles
- Keep line of sight with the drone
- All scripts auto-land on exit or error
- Speed limits are conservative by default
- Person detection triggers immediate backoff in cat mode
