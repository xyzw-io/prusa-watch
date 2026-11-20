//! Host-side YOLOv2 region decoding (`src/region.rs`).

use prusa_watch::region::{self, Layout, OBICO_ANCHORS, VALUES_PER_ANCHOR};

const GRID: usize = 13;
const CHANNELS: usize = OBICO_ANCHORS.len() * VALUES_PER_ANCHOR;

/// A deterministic, non-uniform head tensor in NCHW order.
fn synthetic_nchw() -> Vec<f32> {
    (0..CHANNELS * GRID * GRID)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
        .collect()
}

fn nchw_to_nhwc(nchw: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; nchw.len()];
    for c in 0..CHANNELS {
        for row in 0..GRID {
            for col in 0..GRID {
                out[(row * GRID + col) * CHANNELS + c] = nchw[(c * GRID + row) * GRID + col];
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn output_shape_matches_the_onnx_export() {
    let d = region::decode(&synthetic_nchw(), Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).unwrap();
    assert_eq!(d.confs.len(), 845);
    assert_eq!(d.boxes.len(), 845 * 4);
}

#[test]
fn nchw_and_nhwc_decode_identically() {
    let nchw = synthetic_nchw();
    let a = region::decode(&nchw, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).unwrap();
    let b = region::decode(&nchw_to_nhwc(&nchw), Layout::Nhwc, GRID, GRID, &OBICO_ANCHORS).unwrap();
    assert_eq!(a, b);
}

#[test]
fn one_cell_decodes_to_the_documented_formula() {
    // Anchor 3, row 4, col 9 with known raw values; everything else zero.
    let (a, row, col) = (3usize, 4usize, 9usize);
    let raw_vals = [0.4f32, -1.1, 0.3, -0.2, 2.5, -7.0];
    let mut nchw = vec![0.0f32; CHANNELS * GRID * GRID];
    for (k, v) in raw_vals.iter().enumerate() {
        nchw[((a * VALUES_PER_ANCHOR + k) * GRID + row) * GRID + col] = *v;
    }
    let d = region::decode(&nchw, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).unwrap();
    let i = a * GRID * GRID + row * GRID + col;
    let g = GRID as f32;
    let cx = (sigmoid(0.4) + col as f32) / g;
    let cy = (sigmoid(-1.1) + row as f32) / g;
    let w = 0.3f32.exp() * OBICO_ANCHORS[a][0] / g;
    let h = (-0.2f32).exp() * OBICO_ANCHORS[a][1] / g;
    let expect = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
    for (k, e) in expect.iter().enumerate() {
        assert!(
            (d.boxes[i * 4 + k] - e).abs() < 1e-6,
            "coord {k}: {} vs {e}",
            d.boxes[i * 4 + k]
        );
    }
    // Confidence is sigmoid(objectness) only; the class channel (-7.0) is ignored.
    assert!((d.confs[i] - sigmoid(2.5)).abs() < 1e-7);
    // Order is anchor-major: the neighbouring index belongs to the next column of the same anchor.
    assert!((d.confs[i + 1] - 0.5).abs() < 1e-7);
}

#[test]
fn rejects_wrong_sizes_and_non_finite_values() {
    let good = synthetic_nchw();
    assert!(region::decode(&good[1..], Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).is_err());
    assert!(region::decode(&good, Layout::Nchw, GRID, 12, &OBICO_ANCHORS).is_err());
    assert!(region::decode(&good, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS[..4]).is_err());
    assert!(region::decode(&good, Layout::Nchw, GRID, GRID, &[]).is_err());
    let mut nan = good.clone();
    nan[100] = f32::NAN;
    assert!(region::decode(&nan, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).is_err());
    let mut huge = good;
    huge[2 * GRID * GRID] = 200.0; // tw of anchor 0: exp overflows to infinity
    let err = region::decode(&huge, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).unwrap_err();
    assert!(err.contains("non-finite"), "{err}");
}

/// The decoder against the real Obico model: run tract on the full ONNX with the last convolution
/// (tensor "218") exposed as an extra output, decode that tensor on the host, and compare with the
/// model's own `boxes`/`confs` on the repo's vision fixtures. Needs the model file:
///   PRUSA_WATCH_MODEL=models/model-weights.onnx cargo test --test region -- --ignored
#[test]
#[ignore]
fn decoder_matches_the_onnx_tail_on_real_frames() {
    use tract_onnx::prelude::*;
    let model_path = std::env::var("PRUSA_WATCH_MODEL").expect("requires PRUSA_WATCH_MODEL");
    prusa_watch::model::verify_digest(std::path::Path::new(&model_path), prusa_watch::model::DEFAULT_SHA256).unwrap();
    let mut model = tract_onnx::onnx().model_for_path(&model_path).unwrap();
    model.set_input_fact(0, f32::fact([1, 3, 416, 416]).into()).unwrap();
    model.select_outputs_by_name(["boxes", "confs", "218"]).unwrap();
    let plan = model.into_optimized().unwrap().into_runnable().unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision");
    let (mut worst_box, mut worst_conf) = (0f32, 0f32);
    for name in ["field-1.png", "field-2.png", "field-3.png", "pattern-641x359.png"] {
        let img = image::open(dir.join(name)).unwrap().to_rgb8();
        let resized = prusa_watch::imaging::resize_linear(&img, 416, 416);
        let raw = resized.as_raw();
        let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 416, 416), |(_, c, y, x)| {
            raw[(y * 416 + x) * 3 + c] as f32 / 255.0
        })
        .into();
        let out = plan.run(tvec!(input.into())).unwrap();
        let boxes: Vec<f32> = out[0].to_plain_array_view::<f32>().unwrap().iter().copied().collect();
        let confs: Vec<f32> = out[1].to_plain_array_view::<f32>().unwrap().iter().copied().collect();
        let head = out[2].to_plain_array_view::<f32>().unwrap();
        assert_eq!(head.shape(), &[1, CHANNELS, GRID, GRID]);
        let head: Vec<f32> = head.iter().copied().collect();
        let d = region::decode(&head, Layout::Nchw, GRID, GRID, &OBICO_ANCHORS).unwrap();
        let db = d
            .boxes
            .iter()
            .zip(&boxes)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dc = d
            .confs
            .iter()
            .zip(&confs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let max_conf = confs.iter().copied().fold(0f32, f32::max);
        eprintln!("{name}: max |dbox| {db:.2e}, max |dconf| {dc:.2e}, max conf {max_conf:.3}");
        worst_box = worst_box.max(db);
        worst_conf = worst_conf.max(dc);
    }
    assert!(worst_box < 1e-5, "box mismatch {worst_box}");
    assert!(worst_conf < 1e-6, "confidence mismatch {worst_conf}");
}
