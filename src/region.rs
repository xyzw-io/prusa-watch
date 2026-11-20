//! Host-side decoding of Obico's YOLOv2 "region" head.
//!
//! Obico's failure model is YOLOv2 (Darknet-19): one 13x13 grid, five anchors, one class. Its ONNX
//! export (pytorch-YOLOv4's `yolo_forward`) ends in a decoding tail of Slice/Sigmoid/Exp/Concat ops
//! that turns the last convolution's `[1, 30, 13, 13]` tensor into `boxes [1, 845, 1, 4]` and
//! `confs [1, 845, 1]`. An NPU build of the model stops at that convolution, so this module repeats
//! the tail on the host and hands its result to [`crate::detector::post_process`] unchanged.
//!
//! Per anchor `a`, the 30 channels hold six values `[tx, ty, tw, th, objectness, class]`. The export
//! ignores the class channel (one class), so a box's confidence is `sigmoid(objectness)`:
//!
//! ```text
//! cx = (sigmoid(tx) + col) / grid_w        w = exp(tw) * anchor_w / grid_w
//! cy = (sigmoid(ty) + row) / grid_h        h = exp(th) * anchor_h / grid_h
//! box = [cx - w/2, cy - h/2, cx - w/2 + w, cy - h/2 + h]   (normalized x1, y1, x2, y2)
//! ```
//!
//! Output order is anchor-major (`a * grid_h * grid_w + row * grid_w + col`), as in the export.
//! All arithmetic is float32 in the export's order of operations, so results match the ONNX
//! outputs to rounding (see `tests/region.rs`).

/// Obico model anchors in grid cells (YOLOv2 VOC anchors), read from the ONNX export's constants.
pub const OBICO_ANCHORS: [[f32; 2]; 5] = [
    [1.3221, 1.73145],
    [3.19275, 4.00944],
    [5.05587, 8.09892],
    [9.47112, 4.84053],
    [11.2364, 10.0071],
];

/// Values per anchor in the head: tx, ty, tw, th, objectness, class.
pub const VALUES_PER_ANCHOR: usize = 6;

/// Memory layout of the raw head tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[1, C, H, W]`, as the ONNX graph produces it.
    Nchw,
    /// `[1, H, W, C]`, as Hailo NPUs usually return it.
    Nhwc,
}

/// Decoded boxes (`[N * 4]`, normalized x1, y1, x2, y2) and confidences (`[N]`, one class).
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub boxes: Vec<f32>,
    pub confs: Vec<f32>,
}

/// Same formula as `torch.sigmoid` / the ONNX `Sigmoid` op, in float32.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode a YOLOv2 region head. Errors on a tensor that does not match the grid and anchors, or that
/// holds non-finite values, so a corrupt NPU response can never turn into a "nothing seen" frame.
pub fn decode(
    raw: &[f32],
    layout: Layout,
    grid_h: usize,
    grid_w: usize,
    anchors: &[[f32; 2]],
) -> Result<Decoded, String> {
    let channels = anchors.len() * VALUES_PER_ANCHOR;
    let cells = grid_h * grid_w;
    if anchors.is_empty() || cells == 0 {
        return Err("region head needs at least one anchor and a non-empty grid".into());
    }
    if raw.len() != channels * cells {
        return Err(format!(
            "region head has {} values, expected {} ({} anchors x {} values x {}x{} grid)",
            raw.len(),
            channels * cells,
            anchors.len(),
            VALUES_PER_ANCHOR,
            grid_h,
            grid_w
        ));
    }
    if let Some(bad) = raw.iter().position(|v| !v.is_finite()) {
        return Err(format!("region head value {bad} is not finite"));
    }
    let at = |channel: usize, row: usize, col: usize| -> f32 {
        match layout {
            Layout::Nchw => raw[(channel * grid_h + row) * grid_w + col],
            Layout::Nhwc => raw[(row * grid_w + col) * channels + channel],
        }
    };
    let (gw, gh) = (grid_w as f32, grid_h as f32);
    let n = anchors.len() * cells;
    let mut boxes = Vec::with_capacity(n * 4);
    let mut confs = Vec::with_capacity(n);
    for (a, &[anchor_w, anchor_h]) in anchors.iter().enumerate() {
        let base = a * VALUES_PER_ANCHOR;
        for row in 0..grid_h {
            for col in 0..grid_w {
                // Export order: sigmoid * 1.0 - 0.0, add the grid offset, divide by the grid size.
                let cx = (sigmoid(at(base, row, col)) * 1.0 - 0.0 + col as f32) / gw;
                let cy = (sigmoid(at(base + 1, row, col)) * 1.0 - 0.0 + row as f32) / gh;
                let w = (at(base + 2, row, col).exp() * anchor_w) / gw;
                let h = (at(base + 3, row, col).exp() * anchor_h) / gh;
                let x1 = cx - w * 0.5;
                let y1 = cy - h * 0.5;
                boxes.extend_from_slice(&[x1, y1, x1 + w, y1 + h]);
                confs.push(sigmoid(at(base + 4, row, col)));
            }
        }
    }
    if boxes.iter().chain(&confs).any(|v| !v.is_finite()) {
        return Err("region head decoded to non-finite boxes (exp overflow)".into());
    }
    Ok(Decoded { boxes, confs })
}
