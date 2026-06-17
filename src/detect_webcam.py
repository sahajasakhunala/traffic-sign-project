import cv2
import numpy as np
import torch
import re
from collections import deque, Counter
from torchvision import transforms
from PIL import Image

from model import TrafficSignCNN

# ─────────────────────────────────────────────
#  All 43 GTSRB class names
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Speed limit (20km/h)",        # 0
    "Speed limit (30km/h)",        # 1
    "Speed limit (50km/h)",        # 2
    "Speed limit (60km/h)",        # 3
    "Speed limit (70km/h)",        # 4
    "Speed limit (80km/h)",        # 5
    "End of speed limit (80km/h)", # 6
    "Speed limit (100km/h)",       # 7
    "Speed limit (120km/h)",       # 8
    "No passing",                  # 9
    "No passing (over 3.5t)",      # 10
    "Right-of-way at intersection",# 11
    "Priority road",               # 12
    "Yield",                       # 13
    "Stop",                        # 14
    "No vehicles",                 # 15
    "Vehicles over 3.5t prohib.",  # 16
    "Do not enter",                # 17
    "General caution",             # 18
    "Dangerous curve left",        # 19
    "Dangerous curve right",       # 20
    "Double curve",                # 21
    "Bumpy road",                  # 22
    "Slippery road",               # 23
    "Road narrows on the right",   # 24
    "Road work",                   # 25
    "Traffic signals",             # 26
    "Pedestrians",                 # 27
    "Children crossing",           # 28
    "Bicycles crossing",           # 29
    "Beware of ice/snow",          # 30
    "Wild animals crossing",       # 31
    "End speed + passing limits",  # 32
    "Turn right ahead",            # 33
    "Turn left ahead",             # 34
    "Ahead only",                  # 35
    "Go straight or right",        # 36
    "Go straight or left",         # 37
    "Keep right",                  # 38
    "Keep left",                   # 39
    "Roundabout mandatory",        # 40
    "End of no passing",           # 41
    "End no passing (over 3.5t)",  # 42
]

# ─────────────────────────────────────────────
#  Color-hint → allowed class IDs
#  Red signs:  speed limits, stop, yield,
#              no passing, no entry, no vehicles, etc.
#  Blue signs: mandatory (keep right/left,
#              roundabout, ahead only, go straight)
#  Yellow signs: warning (general caution,
#                curves, bumpy road, road work, etc.)
# ─────────────────────────────────────────────
COLOR_ALLOWED_CLASSES = {
    "red":    {0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,32,41,42},
    "blue":   {33,34,35,36,37,38,39,40},
    "yellow": {11,12,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
}

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_PATH          = "models/traffic_sign_cnn.pth"
IMAGE_SIZE          = 64
CONFIDENCE_THRESH   = 0.50       # lowered to 0.50 for testing new pipeline
MIN_COMPONENT_AREA  = 500
SHAPE_SCORE_THRESH  = 0.55       # tunable multi-metric threshold
SHOW_DEBUG_MASK     = True       # toggle mask debug window

# GTSRB channel stats — must match train.py val_transform
MEAN = [0.3337, 0.3064, 0.3171]
STD  = [0.2672, 0.2564, 0.2629]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
model      = TrafficSignCNN().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Verify class names from checkpoint
if "class_names" in checkpoint:
    print("[INFO] Checkpoint class names (first 20):", checkpoint["class_names"][:20])

# Do NOT overwrite CLASS_NAMES with folder names from checkpoint,
# as we want to keep the human-readable labels.

print(f"[INFO] Loaded  : {MODEL_PATH}")
print(f"[INFO] Best val: {checkpoint.get('val_acc', 'n/a')}")
print(f"[INFO] Epoch   : {checkpoint.get('epoch',   'n/a')}")
print(f"[INFO] Device  : {device}")


# ═════════════════════════════════════════════
#  PHASE A — Detection Quality
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
#  Step 1: Separate color masks
# ─────────────────────────────────────────────
def build_color_masks(hsv):
    """
    Returns a dict of independent binary masks:
      {'red': mask, 'blue': mask, 'yellow': mask}
    Each processed with morphology independently.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    red = (cv2.inRange(hsv, np.array([  0, 100,  70]), np.array([ 10, 255, 255])) |
           cv2.inRange(hsv, np.array([160, 100,  70]), np.array([180, 255, 255])))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  kernel)

    blue = cv2.inRange(hsv, np.array([100, 100,  50]), np.array([140, 255, 255]))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN,  kernel)

    yellow = cv2.inRange(hsv, np.array([ 15,  80,  80]), np.array([ 35, 255, 255]))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,  kernel)

    return {"red": red, "blue": blue, "yellow": yellow}


# ─────────────────────────────────────────────
#  Step 3: Multi-metric shape scoring
# ─────────────────────────────────────────────
def compute_shape_score(component_mask):
    """
    Compute a combined geometric quality score from:
      circularity, solidity, aspect ratio.
    Returns (score, circularity, solidity, aspect).
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, 0.0

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    # Circularity: 4πA / P²  (1.0 = perfect circle)
    circularity = 4 * np.pi * area / (peri ** 2 + 1e-6)
    circularity = min(circularity, 1.0)

    # Solidity: area / convex hull area
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity  = area / (hull_area + 1e-6)
    solidity  = min(solidity, 1.0)

    # Aspect ratio score: 1.0 when square
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = min(w, h) / (max(w, h) + 1e-6)

    score = 0.4 * circularity + 0.3 * solidity + 0.3 * aspect
    return score, circularity, solidity, aspect


# ─────────────────────────────────────────────
#  Step 7 (partial): ROI weighting
# ─────────────────────────────────────────────
def roi_weight(cy, frame_h):
    """
    Returns 1.0 for upper portion of frame,
    decays toward bottom. Floor at 0.7.
    """
    y_norm = cy / frame_h
    if y_norm < 0.65:
        return 1.0
    else:
        return max(0.7, 1.0 - (y_norm - 0.65) * 2.0)


# ─────────────────────────────────────────────
#  Step 2: Extract candidates via connected components
# ─────────────────────────────────────────────
def extract_candidates(color_masks, frame):
    """
    For each color mask, run connectedComponentsWithStats,
    score each component, and return candidate crops tagged
    with their color_hint.

    Returns: [(x1, y1, x2, y2, crop, color_hint), ...]
    """
    h, w = frame.shape[:2]
    candidates = []

    for color_name, mask in color_masks.items():
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_COMPONENT_AREA:
                continue

            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            # Aspect ratio filter (reject very elongated blobs)
            if not (0.5 < bw / (bh + 1e-6) < 2.0):
                continue

            # Extract the component mask for shape scoring
            component_mask = np.uint8(labels == i) * 255
            score, circ, sol, asp = compute_shape_score(component_mask)

            # Apply ROI weight
            weight = roi_weight(cy, h)
            weighted_score = score * weight

            if weighted_score < SHAPE_SCORE_THRESH:
                # Log rejected candidates for debugging
                print(f"  [REJECT] color={color_name} score={score:.2f} "
                      f"(w={weight:.2f} -> {weighted_score:.2f}) "
                      f"circ={circ:.2f} sol={sol:.2f} asp={asp:.2f}")
                continue

            # Log accepted candidates
            print(f"  [ACCEPT] color={color_name} score={score:.2f} "
                  f"(w={weight:.2f} -> {weighted_score:.2f}) "
                  f"circ={circ:.2f} sol={sol:.2f} asp={asp:.2f}")

            # Crop with padding
            pad = 10
            x1, y1 = max(0, bx - pad),      max(0, by - pad)
            x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            candidates.append((x1, y1, x2, y2, crop, color_name))

    return candidates


# ═════════════════════════════════════════════
#  CNN — Batch inference
# ═════════════════════════════════════════════

def classify_batch(crops_bgr: list) -> list[tuple[int, float]]:
    """
    Takes a list of BGR crops, returns [(class_id, confidence), ...].
    Single forward pass.
    """
    tensors = []
    for crop in crops_bgr:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(transform(pil))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs      = torch.softmax(model(batch), dim=1)
        confs, ids = torch.max(probs, dim=1)

    return list(zip(ids.tolist(), confs.tolist()))


# ─────────────────────────────────────────────
#  Step 7: Color-hint validation
# ─────────────────────────────────────────────
def is_color_consistent(class_id, color_hint):
    """
    Check if the CNN prediction is consistent with
    the color channel the candidate originated from.
    """
    allowed = COLOR_ALLOWED_CLASSES.get(color_hint)
    if allowed is None:
        return True  # unknown color, allow everything
    return class_id in allowed


# ═════════════════════════════════════════════
#  PHASE B — Temporal Intelligence
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
#  Step 4 & 5: Tracked sign with persistence
#              and majority voting
# ─────────────────────────────────────────────
class TrackedSign:
    def __init__(self, bbox, label, confidence, color_hint):
        self.bbox       = bbox        # (x1, y1, x2, y2)
        self.label      = label
        self.confidence = confidence
        self.color_hint = color_hint
        self.missed     = 0           # consecutive missed frames
        self.history    = deque(maxlen=5)
        self.history.append(label)

    def update(self, bbox, label, confidence):
        self.bbox       = bbox
        self.label      = label
        self.confidence = confidence
        self.missed     = 0
        self.history.append(label)

    def mark_missed(self):
        self.missed += 1

    @property
    def alive(self):
        return self.missed < 10

    @property
    def voted_label(self):
        """Majority vote across last 5 classifications."""
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]


def compute_iou(box_a, box_b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# Global tracker list
tracked_signs: list[TrackedSign] = []

IOU_MATCH_THRESH = 0.3


def update_tracking(raw_detections):
    """
    Match raw detections to existing tracked signs via IoU.
    Create new trackers for unmatched detections.
    Mark unmatched trackers as missed.
    Returns list of active tracked signs for display.
    """
    global tracked_signs

    # Mark all existing trackers as potentially missed this frame
    matched_tracker_idx = set()
    matched_det_idx     = set()

    # Greedy matching: for each detection, find best overlapping tracker
    for d_idx, (x1, y1, x2, y2, label, conf, color_hint) in enumerate(raw_detections):
        det_box  = (x1, y1, x2, y2)
        best_iou = 0.0
        best_t   = -1

        for t_idx, tracker in enumerate(tracked_signs):
            if t_idx in matched_tracker_idx:
                continue
            iou = compute_iou(det_box, tracker.bbox)
            if iou > best_iou:
                best_iou = iou
                best_t   = t_idx

        if best_iou >= IOU_MATCH_THRESH and best_t >= 0:
            # Update existing tracker
            tracked_signs[best_t].update(det_box, label, conf)
            matched_tracker_idx.add(best_t)
            matched_det_idx.add(d_idx)
        else:
            # New tracker
            det_box = (x1, y1, x2, y2)
            tracked_signs.append(TrackedSign(det_box, label, conf, color_hint))
            matched_det_idx.add(d_idx)

    # Mark unmatched trackers as missed
    for t_idx in range(len(tracked_signs)):
        if t_idx not in matched_tracker_idx:
            tracked_signs[t_idx].mark_missed()

    # Prune dead trackers
    tracked_signs = [t for t in tracked_signs if t.alive]

    # Build display list using voted labels
    display = []
    for t in tracked_signs:
        x1, y1, x2, y2 = t.bbox
        display.append((x1, y1, x2, y2, t.voted_label, t.confidence))

    return display


# ═════════════════════════════════════════════
#  PHASE C — Driving Logic
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
#  Step 6: Speed memory
# ─────────────────────────────────────────────
def extract_speed_limit(label):
    """
    Only extract speed limits from labels that
    actually start with 'Speed limit'.
    """
    if not label.startswith("Speed limit"):
        return None

    match = re.search(r"Speed limit \((\d+)km/h\)", label)
    if match:
        return int(match.group(1))
    return None


class SpeedMemory:
    def __init__(self):
        self.current_limit = None

    def update(self, label):
        """
        Update speed memory based on detected sign label.
        - Speed limit signs set the limit.
        - End-of-speed-limit signs clear it.
        - All other signs are ignored.
        """
        speed = extract_speed_limit(label)
        if speed is not None:
            self.current_limit = speed
            return

        # "End of speed limit" or "End speed + passing limits" clears memory
        if label.startswith("End of speed limit") or label.startswith("End speed"):
            self.current_limit = None


speed_memory = SpeedMemory()


# ═════════════════════════════════════════════
#  Main detection pipeline
# ═════════════════════════════════════════════

def detect_signs(frame):
    """
    Full pipeline:
      Color Segmentation → Connected Components →
      Shape Scoring → CNN → Color-Hint Validation
    Returns raw detections (before tracking).
    """
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Step 1: Separate color masks
    color_masks = build_color_masks(hsv)

    # Debug: show combined mask
    if SHOW_DEBUG_MASK:
        combined = color_masks["red"] | color_masks["blue"] | color_masks["yellow"]
        cv2.imshow("Mask", combined)

    # Step 2 + 3: Extract candidates via connected components + shape scoring
    candidates = extract_candidates(color_masks, frame)

    if not candidates:
        return []

    # Batch CNN inference
    crops   = [c[4] for c in candidates]
    results = classify_batch(crops)

    # Apply confidence threshold + color-hint validation (Step 7)
    detections = []
    for (x1, y1, x2, y2, _, color_hint), (class_id, confidence) in zip(candidates, results):
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
        print(f"Pred: {class_id} ({label}) Conf: {confidence:.4f}")
        if confidence < CONFIDENCE_THRESH:
            continue

        # Color-hint sanity check
        if not is_color_consistent(class_id, color_hint):
            print(f"  [COLOR REJECT] {label} from {color_hint} mask - inconsistent")
            continue

        detections.append((x1, y1, x2, y2, label, confidence, color_hint))

    return detections


# ─────────────────────────────────────────────
#  Draw
# ─────────────────────────────────────────────
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, label, conf = det[:6]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


# ─────────────────────────────────────────────
#  Main webcam loop
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    print("[INFO] Press Q to quit.")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESH}")
    print(f"[INFO] Shape score threshold: {SHAPE_SCORE_THRESH}")
    print(f"[INFO] Debug mask: {'ON' if SHOW_DEBUG_MASK else 'OFF'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Phase A: detect candidates + CNN
        raw_detections = detect_signs(frame)

        # Phase B: tracking + majority voting
        tracked_display = update_tracking(raw_detections)

        # Phase C: update speed memory from tracked (voted) labels
        for _, _, _, _, label, _ in tracked_display:
            speed_memory.update(label)

        # Draw tracked detections
        frame = draw_detections(frame, tracked_display)

        # HUD overlay
        cv2.putText(frame, f"Detections: {len(tracked_display)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Tracked: {len(tracked_signs)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESH*100:.0f}%", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if speed_memory.current_limit is not None:
            cv2.putText(
                frame,
                f"Current Limit: {speed_memory.current_limit} km/h",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Traffic Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()