# main.py
# Integrated lane detection: BEV + LDW + webcam (DirectShow backend)
# Requirements: pip install opencv-python numpy

import cv2
import numpy as np
import argparse
import time
from collections import deque

# ---------- CONFIG ----------
WHITE_LIGHTNESS_LOWER = 200
YELLOW_HUE_LOWER = 15
YELLOW_HUE_UPPER = 35

# Perspective warp source polygon as ratios of width/height (tunable)
SRC_REL = np.array([
    [0.15, 0.95],   # bottom-left
    [0.45, 0.60],   # mid-left (apex left)
    [0.55, 0.60],   # mid-right (apex right)
    [0.95, 0.95]    # bottom-right
], dtype=np.float32)

DST_REL = np.array([
    [0.2, 1.0],
    [0.2, 0.0],
    [0.8, 0.0],
    [0.8, 1.0]
], dtype=np.float32)

SMOOTHING_ALPHA = 0.25
SMOOTHING_HISTORY = 8

LDW_OFFSET_PIXELS = 80
LDW_SHOW_WARNING = True

M_PER_PIXEL = 3.7 / 300.0
# ----------------------------

left_fit_history = deque(maxlen=SMOOTHING_HISTORY)
right_fit_history = deque(maxlen=SMOOTHING_HISTORY)

def color_mask(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, WHITE_LIGHTNESS_LOWER, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([YELLOW_HUE_LOWER, 90, 100], dtype=np.uint8)
    upper_yellow = np.array([YELLOW_HUE_UPPER, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return combined

def bottom_half_only(img):
    mask = np.zeros_like(img)
    h = img.shape[0]
    mask[int(h*0.5):, :] = 255
    return cv2.bitwise_and(img, mask)

def region_of_interest(img):
    h, w = img.shape[:2]
    apex_y = int(h * 0.60)
    triangle = np.array([
        [(int(0.12*w), h), (int(0.88*w), h), (int(0.5*w), apex_y)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(img, mask)

def get_perspective_transform_matrices(frame_shape):
    h, w = frame_shape[0], frame_shape[1]
    src = np.zeros((4,2), dtype=np.float32)
    dst = np.zeros((4,2), dtype=np.float32)
    for i in range(4):
        src[i] = (SRC_REL[i][0] * w, SRC_REL[i][1] * h)
        dst[i] = (DST_REL[i][0] * w, DST_REL[i][1] * h)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst

def warp_image(img, M, dst_size):
    return cv2.warpPerspective(img, M, dst_size, flags=cv2.INTER_LINEAR)

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    minpix = 50

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    if len(left_lane_inds) == 0:
        left_lane_inds = np.array([], dtype=int)
    else:
        left_lane_inds = np.concatenate(left_lane_inds)
    if len(right_lane_inds) == 0:
        right_lane_inds = np.array([], dtype=int)
    else:
        right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds] if left_lane_inds.size else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size else np.array([])

    return leftx, lefty, rightx, righty

def fit_polynomial(leftx, lefty, rightx, righty, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fit = None
    right_fit = None
    if leftx.size > 0 and lefty.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if rightx.size > 0 and righty.size > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, ploty

def make_lane_overlay(frame, left_fit, right_fit, ploty, Minv):
    warp_zero = np.zeros_like(frame[:, :, 0]).astype(np.uint8)
    color_warp = np.zeros_like(frame).astype(np.uint8)
    h, w = frame.shape[0], frame.shape[1]

    if left_fit is None and right_fit is None:
        return color_warp

    left_pts = None
    right_pts = None

    if left_fit is not None:
        leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        left_pts = np.array([np.transpose(np.vstack([leftx, ploty]))]).astype(np.int32)

    if right_fit is not None:
        rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        right_pts = np.array([np.transpose(np.vstack([rightx, ploty]))]).astype(np.int32)

    if (left_pts is not None and left_pts.size > 0) and (right_pts is not None and right_pts.size > 0):
        pts = np.hstack((left_pts, np.flip(right_pts, axis=1)))
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    else:
        if left_pts is not None and left_pts.size > 0:
            lp = left_pts[0]
            for i in range(len(lp) - 1):
                cv2.line(color_warp, tuple(lp[i]), tuple(lp[i + 1]), (0, 255, 0), 8)
        if right_pts is not None and right_pts.size > 0:
            rp = right_pts[0]
            for i in range(len(rp) - 1):
                cv2.line(color_warp, tuple(rp[i]), tuple(rp[i + 1]), (0, 255, 0), 8)

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    return newwarp

def smooth_fit_history(left_fit, right_fit):
    if left_fit is not None:
        left_fit_history.append(left_fit)
    if right_fit is not None:
        right_fit_history.append(right_fit)

    avg_left = None
    avg_right = None
    if len(left_fit_history) > 0:
        avg_left = np.mean(left_fit_history, axis=0)
    if len(right_fit_history) > 0:
        avg_right = np.mean(right_fit_history, axis=0)
    return avg_left, avg_right

def compute_lane_center_offset(left_fit, right_fit, img_shape):
    h, w = img_shape[0], img_shape[1]
    y_eval = h - 1
    if left_fit is not None and right_fit is not None:
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2.0
        veh_center = w / 2.0
        offset_pixels = veh_center - lane_center
        offset_meters = offset_pixels * M_PER_PIXEL
        return offset_pixels, offset_meters, lane_center
    if left_fit is not None:
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = left_x + 300
        lane_center = (left_x + right_x) / 2.0
        veh_center = w / 2.0
        offset_pixels = veh_center - lane_center
        offset_meters = offset_pixels * M_PER_PIXEL
        return offset_pixels, offset_meters, lane_center
    if right_fit is not None:
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        left_x = right_x - 300
        lane_center = (left_x + right_x) / 2.0
        veh_center = w / 2.0
        offset_pixels = veh_center - lane_center
        offset_meters = offset_pixels * M_PER_PIXEL
        return offset_pixels, offset_meters, lane_center
    return None, None, None

def overlay_ldw_text(img, offset_pixels, offset_meters):
    h, w = img.shape[0], img.shape[1]
    if offset_pixels is None:
        return img
    txt = f"Offset: {offset_meters:.2f} m ({int(offset_pixels)} px)"
    cv2.putText(img, txt, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    if abs(offset_pixels) > LDW_OFFSET_PIXELS and LDW_SHOW_WARNING:
        warn_text = "LANE DEPARTURE RISK!"
        cv2.putText(img, warn_text, (int(w*0.1), int(h*0.3)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 4)
    return img

def process_frame(frame, M, Minv, dst_size):
    frame = cv2.resize(frame, (dst_size[0], dst_size[1]))
    h, w = frame.shape[0], frame.shape[1]

    mask = color_mask(frame)
    mask = bottom_half_only(mask)
    masked_color = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    warped_edges = warp_image(edges, M, dst_size)
    _, binary_warped = cv2.threshold(warped_edges, 50, 255, cv2.THRESH_BINARY)

    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    left_fit, right_fit, ploty = fit_polynomial(leftx, lefty, rightx, righty, binary_warped)

    avg_left, avg_right = smooth_fit_history(left_fit, right_fit)
    overlay = make_lane_overlay(frame, avg_left, avg_right, ploty, Minv)

    offset_pixels, offset_meters, lane_center = compute_lane_center_offset(avg_left, avg_right, frame.shape)
    result = cv2.addWeighted(frame, 1.0, overlay, 0.7, 0.0)
    result = overlay_ldw_text(result, offset_pixels, offset_meters)

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="use webcam instead of video")
    ap.add_argument("--video", type=str, default="video.mp4", help="input video file (if not webcam)")
    args = ap.parse_args()

    if args.webcam:
        # use DirectShow backend for Windows webcams (more reliable)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        use_video_writer = False
    else:
        cap = cv2.VideoCapture(args.video)
        use_video_writer = True

    ret, frame = cap.read()
    if not ret:
        print("ERROR: cannot read from input (video/webcam).")
        return

    h0, w0 = frame.shape[0], frame.shape[1]
    dst_w, dst_h = w0, h0
    M, Minv, src_pts, dst_pts = get_perspective_transform_matrices(frame.shape)
    print("Perspective SRC points (absolute):\n", src_pts)
    print("Perspective DST points (absolute):\n", dst_pts)

    if use_video_writer:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (dst_w, dst_h))
    else:
        out = None

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, M, Minv, (dst_w, dst_h))
        cv2.imshow("Lane Detection (BEV + LDW)", result)
        if use_video_writer:
            out.write(result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("snapshot_output.png", result)
            print("Saved snapshot_output.png")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
