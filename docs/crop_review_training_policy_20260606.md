# Crop Review And Training Policy 2026-06-06

## Regression Snapshot

- Dataset: `stage5_mobile_results/uploads`, 37 uploaded shelf images.
- Output: `stage5_mobile_results/regression_20260606_thin_boundary_uploads37`.
- Result summary:
  - 37 images total.
  - 6 images skipped as `bad_position`.
  - 17 images contained yellow review items.
  - 28 yellow items total.
  - 1 image contained red items, 2 red items total.
- Thin-book rule result:
  - Thin-book inferred gapfill appeared in 4 yellow items across 37 images.
  - No broad explosion of false gapfill boxes was observed in this regression.

## Crop Training Use Rules

Use a crop for recognition training only when all of these are true:

- The crop contains one complete call number.
- The call number is visually readable by a human.
- The crop is not dominated by red label occlusion, blur, perspective distortion, or cut-off text.
- The label has been manually confirmed.

Do not directly train these categories:

- `疑似漏检补框`: generated from adjacent call-number gaps or thin-book number inference. Keep for human review only unless the crop is manually confirmed to contain exactly one complete readable call number.
- `OCR 未识别`: keep out of training unless the crop is complete and clear, then manually label first.
- `低置信 OCR`: train only if the crop is complete, clear, and manually corrected.
- `前缀/排架差异`: do not train until the shelf order and call number prefix are manually confirmed.
- `bad_position`: do not train. Ask the user to retake the photo.
- Red order items: do not use for recognition training until the OCR text itself is verified.

## Recommended Next Training Gate

Before the next 2-epoch recognition training run:

1. Export only manually confirmed, complete, readable crops.
2. Exclude all inferred yellow gapfill boxes by default.
3. Exclude red-label-obscured or cropped-off call numbers.
4. Keep official OCR as the Web default until full-image comparison beats it on detection count, green count, yellow count, and speed.
