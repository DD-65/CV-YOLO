from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolo26s-pose.pt")
cap = cv2.VideoCapture(0)

CONF_TH = 0.25
EMA_ALPHA = 0.20
CALIB_SECS = 4.0
SLOUCH_RATIO = 0.78
WARN_AFTER_SECS = 2.0

ema_head = None
ema_forward = None
baseline_head = None
baseline_forward = None
calib_head = []
calib_forward = []
slouch_since = None
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model(frame, verbose=False)[0]
    vis = r.plot()
    msg = "No person detected"
    color = (255, 255, 255)

    if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))

        kxy = r.keypoints.xy[i].cpu().numpy()
        if r.keypoints.conf is not None:
            kcf = r.keypoints.conf[i].cpu().numpy()
        else:
            kcf = np.ones(17, dtype=np.float32)

        # COCO keypoints: nose=0, l_shoulder=5, r_shoulder=6.
        need = [0, 5, 6]
        if np.all(kcf[need] > CONF_TH):
            nose = kxy[0]
            l_sh, r_sh = kxy[5], kxy[6]

            neck = (l_sh + r_sh) / 2.0
            shoulder_w = np.linalg.norm(l_sh - r_sh) + 1e-6

            # Larger head_height means head is held higher relative to shoulders.
            head_height = (neck[1] - nose[1]) / shoulder_w
            # Larger forward_offset means head moved sideward/forward from shoulder center.
            forward_offset = abs(nose[0] - neck[0]) / shoulder_w

            if ema_head is None:
                ema_head = head_height
                ema_forward = forward_offset
            else:
                ema_head = (1 - EMA_ALPHA) * ema_head + EMA_ALPHA * head_height
                ema_forward = (1 - EMA_ALPHA) * ema_forward + EMA_ALPHA * forward_offset

            if baseline_head is None:
                calib_head.append(ema_head)
                calib_forward.append(ema_forward)
                msg = "Calibrating... sit upright"
                color = (0, 255, 255)
                if (time.time() - t0) >= CALIB_SECS and len(calib_head) >= 15:
                    baseline_head = float(np.median(calib_head))
                    baseline_forward = float(np.median(calib_forward))
            else:
                slouch = (
                    ema_head < (SLOUCH_RATIO * baseline_head)
                    or ema_forward > (baseline_forward + 0.22)
                )
                if slouch:
                    if slouch_since is None:
                        slouch_since = time.time()
                else:
                    slouch_since = None

                if slouch_since is not None and (time.time() - slouch_since) >= WARN_AFTER_SECS:
                    msg = "Warning: Sit upright"
                    color = (0, 0, 255)
                else:
                    msg = "Posture OK"
                    color = (0, 200, 0)
        else:
            msg = "Pose unclear (need nose + both shoulders)"
            color = (0, 165, 255)

    cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.imshow("Posture Coach", vis)

    if (cv2.waitKey(1) & 0xFF) == 27:
        break

cap.release()
cv2.destroyAllWindows()
