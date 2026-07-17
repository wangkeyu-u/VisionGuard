# VisionGuard Dataset Card

## Summary

VisionGuard uses a cleaned derivative of the Roboflow Universe dataset **Construction PPE v3**, whose included metadata identifies it as CC BY 4.0. The original export contained 8,845 640×640 images in YOLO format. This repository does not redistribute the images or labels.

- Source: `https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo`
- Recorded export time: 2024-10-07 11:09 GMT
- Local preparation time: 2026-07-14
- Frozen dataset SHA-256: `6d40b6e5d09cc4e1ed3a6d8d9a8a88259f1e18dc9f117d664fa5694473db3bb2`

Users should verify the current source page, attribution requirements, and rights to any identifiable imagery before redistribution or deployment.

## Final taxonomy and distribution

The final dataset has 8,762 images and 43,727 annotations.

| Class | Annotations | Images containing class |
| --- | ---: | ---: |
| person | 5,738 | 4,765 |
| helmet | 8,379 | 6,468 |
| vest | 10,385 | 7,655 |
| gloves | 5,317 | 2,830 |
| boots | 11,540 | 5,470 |
| no_helmet | 994 | 422 |
| no_vest | 1,374 | 601 |

| Split | Images | Annotations | Target ratio |
| --- | ---: | ---: | ---: |
| Train | 6,125 | 30,656 | 70% |
| Validation | 1,753 | 8,747 | 20% |
| Test | 884 | 4,324 | 10% |

## Preparation

1. Merge synonymous class labels into a canonical PPE taxonomy.
2. Remove the unrelated `glasses` class.
3. Remove `no_gloves` and `no_boots`, which had only 1 and 9 source annotations.
4. Remove five images left without retained annotations.
5. Hash image bytes with SHA-256 and remove 78 exact duplicates.
6. Group images by the filename prefix before Roboflow's `.rf.` suffix.
7. Assign whole groups using approximate multilabel stratification with seed 42.
8. Audit image/label pairs, class IDs, hashes, duplicate leakage, and group leakage.
9. Write a freeze manifest containing every image and label hash.

The final audit found zero exact duplicate groups, zero cross-split SHA-256 groups, and zero cross-split filename-derived source groups.

## Known limitations

- Filename grouping is only a proxy for the original video, site, or capture session. Semantically similar frames can still cross splits if their names do not expose their relationship.
- Of the 78 duplicate groups, 52 contained inconsistent label content. The retained member was selected by most retained objects and then lexicographic path—not by human adjudication.
- `no_helmet` and `no_vest` are minority classes. The test split contains them in only 42 and 60 images, respectively.
- The source combines unknown locations, cameras, capture conditions, and annotation practices. Demographic and geographic coverage has not been audited.
- Images may contain identifiable workers. No privacy or consent audit has been performed.
- The prepared set contains annotated objects; it is not a representative estimate of real-world violation prevalence.
- The upstream export reports stretch resizing to 640×640, which can alter object geometry.

## Appropriate use

Suitable for coursework, reproducible object-detection experiments, pipeline engineering, and human-reviewed prototype demonstrations. It is not sufficient by itself for automated enforcement, worker discipline, access control, or safety certification.
