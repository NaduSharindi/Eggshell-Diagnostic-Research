# Eggshell Candling Image Dataset

Candling (transillumination) images of breeder eggs, annotated for micro-crack
presence and crack morphology. Collected for the final-year project
*A Multi-Modal Deep Learning Approach for Eggshell Micro-Crack Morphology Analysis and Root-Cause Diagnosis in Poultry Breeder Farms* 
(M.P.N. Sharindi, Coventry index 16110429, NIBM Index COBSCCOMP242P-063).

## Summary

| Item | Value |
| --- | --- |
| Eggs | 652 (one image per egg, E001 to E652) |
| Capture date | 12 January 2026 |
| Capture device | Smartphone camera above a candling light inside a light-proof box |
| Original resolution | 3672 x 8160 px (JPEG) |
| Processed images | Square crops around the egg with a 10% margin |
| Annotation tool | Label Studio 1.23.0 |
| Annotation guideline | `annotation/Eggshell_Micro-Crack_Annotation_Guideline` |

## Labels

| Label | Eggs |
| --- | --- |
| Intact | 316 |
| Micro_cracked | 336 |
| - Hairline | _fill in_ |
| - Star | _fill in_ |
| - Network | _fill in_ |

Every Micro_cracked egg also has a polygon outline of the crack region, used to
evaluate Grad-CAM heatmaps.

## Folder contents

| Path | Contents | In Git? |
| --- | --- | --- |
| `annotations/final_labels_v2.json` | Label Studio JSON-MIN export (status, pattern, polygons) | Yes |
| `annotations/final_labels_v2.csv` | Same labels as CSV | Yes |
| `metadata/mapping.csv` | Egg_ID to original camera filename and capture time | Yes |
| `metadata/crop_log.csv` | Automatic crop results per image | Yes |
| `metadata/review_log.csv` | Manual crop review decisions (OK / PARTIAL / EXCLUDED) | Yes |
| `images/` | Cropped images `E001.jpg` ... `E652.jpg` | No (see below) |

## Where the images are stored

Images are not stored in this repository because of their size (raw set about
3 GB, cropped set about 330 MB) and farm data-sharing conditions.

- Raw images (`00_raw_backup`): _add private storage location_
- Cropped images (`02_cropped`): _add private storage location_

Access is limited to the researcher and supervisor.

## Processing steps

1. Raw images backed up unchanged (`00_raw_backup`).
2. Files renamed to Egg_IDs in capture order (`mapping.csv`).
3. Automatic egg detection and square cropping (`src/data_preparation/crop_eggs.py`).
4. Images the script could not crop confidently reviewed by hand
   (`src/data_preparation/review_crop.py`, `review_log.csv`).
5. Annotation in Label Studio following the guideline, including a 30-image pilot
   and a second review pass while outlining cracks (31 labels corrected).

## Pending

- Breeder age class per egg, from farm flock records (pseudonymised).
- Self-consistency check (30 re-labelled eggs) and expert verification.

## Ethics and confidentiality

Farm and flock details are pseudonymised in line with the collaborating
organisation's conditions. No company names or raw operational figures are stored
in this repository.
