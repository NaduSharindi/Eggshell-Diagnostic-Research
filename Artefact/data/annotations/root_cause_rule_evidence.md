# Evidence Base for the Root-Cause Rule Table

**Purpose:** This document is the audit trail behind `root_cause_rules.csv`. It exists so that the physiological/mechanical percentages in the dissertation can be defended as derived from breeder-management documentation and peer-reviewed literature rather than asserted. It becomes **Appendix G** and is summarised in **Section 4.6.4**.

**Status:** v1.0. Requires the hatchery manager's signed attestation (see §5).

---

## 1. The rule table

Physiological-nutritional % / mechanical-environmental %, by crack morphology and Ross 308 lay stage.

| Morphology | Early lay (<32 wk) | Peak production (32–50 wk) | Late lay (>50 wk) |
|---|---|---|---|
| **Hairline** | 30 / 70 | 40 / 60 | 55 / 45 |
| **Network** | 50 / 50 | 60 / 40 | 75 / 25 |
| **Star** | 15 / 85 | 20 / 80 | 30 / 70 |

Two gradients are encoded, and both must be evidenced:

* **Across columns** — the physiological share rises with flock age for every morphology.
* **Down rows** — at any given age, Star is the most mechanical, Network the most physiological, Hairline intermediate.

---

## 2. Evidence for the age gradient (left to right)

The claim is that shell integrity declines with flock age, so a crack of any given morphology becomes progressively more attributable to shell weakness rather than to handling force.

| # | Claim | Source | What it establishes |
|---|---|---|---|
| E1 | Shell thickness declines and egg size increases as the hen ages, with no proportionate increase in shell weight | Aviagen (2020); Aviagen (2020b) | The primary-breeder's own technical note lists hen age as a direct influence on shell quality |
| E2 | Shell breaking strength and thickness decrease significantly across peak, standard-cycle and extended-cycle ages | Nys et al. (2023) | Mechanical quantification of the decline, not just thickness |
| E3 | Age-related deterioration is driven by changes in shell ultrastructure (effective thickness, grain morphology, crystal texture) regulated in the uterus | Zhang et al. (2020) | Mechanistic basis: the decline is structural, not simply a calcium-supply failure |
| E4 | Both the efficiency of calcium absorption from the gut and skeletal calcium resorption decline with age | Aviagen (2020); Leeson & Summers (2000, as cited in Aviagen, 2020) | Physiological mechanism underpinning the gradient |
| E5 | Shell quality is lower in broiler breeders than in commercial layers and deteriorates further under elevated temperature | Ebeid et al. (2014) | Confirms the gradient applies specifically to broiler breeders, the population studied here |
| E6 | Ross 308 parent-stock calcium specification rises across lay: Breeder 1 = 3.00 %, Breeder 2 = 3.20 %, Breeder 3 = 3.40 % | Aviagen (2021) | **The breed standard itself encodes an age gradient in calcium demand.** This is the strongest single justification for using Ross 308 lay stage as the metadata variable, and it is why the age bands are defined by feed phase rather than arbitrary week cut-offs |

> **Design note.** E6 is the reason the `age_class` variable is defined as Early lay / Peak production / Late lay rather than as continuous weeks: the farm's own feeding programme is already discretised on these boundaries, so the metadata is operationally available at inference time without extra record-keeping.

---

## 3. Evidence for the morphology gradient (top to bottom)

| # | Claim | Source | Effect on the rule |
|---|---|---|---|
| E7 | Hairline cracks typically occur when an egg strikes an inflexible surface; star cracks are more likely to result from egg-to-egg collision | Gupta (2008, as cited in Perić et al., 2023) | Both morphologies have a mechanical trigger, but star cracks require a discrete collision event |
| E8 | Impact of sufficient force produces multiple cracks propagating radially from the point of impact (gross and star cracks); weak mechanical force produces a minute hairline artefact | Priyadarshini et al. (2017) | **Star = high-energy point impact.** This is why Star carries the highest mechanical share (85 % at early lay) in the rule table |
| E9 | Hairline cracks are fine fractures usually running lengthwise along the shell; star cracks are slightly indented cracks radiating from a central impact point | Lee et al. (2017), citing Arivazhagan et al. | Operational definitions used verbatim in the annotation guideline |
| E10 | Listed causes of hairline cracks include **reduced shell strength** as well as mechanical damage from beaks, toenails, rough handling and infrequent collection | *Optimum egg quality: A practical approach* (ThePoultrySite, n.d.) | Hairline is the morphology with a **documented dual aetiology**, justifying its intermediate position and its steep age gradient (30 % → 55 %) |
| E11 | Hairline-cracked eggs lose significantly more weight during incubation and suffer markedly higher embryonic mortality than star-cracked eggs (51.1 % vs 36.6 %) | Khabisi et al. (2012) | Establishes that the morphologies are not interchangeable — they differ in biological consequence, which is the practical reason classifying them matters |
| E12 | Up to 2 % of broiler-breeder hatching eggs are rejected for cracked or broken shells; damage arises in the laying nest, during collection and during storage | Perić et al. (2023) | Supplies the mechanical-environmental causal pathways named in the `suggested_actions` text (nest, belts, trays, collisions) |

---

## 4. Evidence for the recommended actions

The `suggested_actions` strings must each trace to a source. Current coverage:

| Action text | Source |
|---|---|
| "Move to Breeder 3 calcium (about 3.4 %)" | Aviagen (2021) — Breeder 3 specification is exactly 3.40 % |
| "Add 2–4 mm limestone or oyster shell" (coarse particulate calcium) | Aviagen (2020); Reis et al. (1995, as cited in ThePoultrySite, n.d.) |
| "Feed in the late afternoon" | Farmer et al. (1983, as cited in ThePoultrySite, n.d.) — shell deposition peaks overnight, so afternoon feeding aligns calcium availability with calcification |
| "Check vitamin D3" | Aviagen (2020) — minimum 3,500 IU/kg recommended for shell quality and hatchability |
| "Reduce heat stress" | Ebeid et al. (2014); Aviagen (2020) |
| Nest, belt, roll-out angle and tray handling changes | ThePoultrySite (n.d.) — specifies 7–8° roll-out angles, cushioned egg guides, ≤6 filler stacking, collection at least twice daily |
| "Do not set cracked eggs" / remove from incubation | Khabisi et al. (2012); Boerjan (ThePoultrySite, n.d.) |

---

## 5. Honest limitations — these go in Section 6.2.3

Three things must be stated plainly rather than discovered by an examiner.

1. **The percentages are expert-elicited point estimates, not measured posterior probabilities.** No study reports P(physiological cause | morphology, age) for broiler breeders. The literature establishes the *direction and ordering* of the gradients; the specific values were set by the hatchery manager within those constraints. The dissertation claims the rule base is **evidence-consistent and internally coherent**, not empirically validated.

2. **"Network" is the weakest-grounded class.** Hairline and star cracks are standard industry terminology with clear literature definitions (E7–E11). Widespread fine crazing is recognised in practice but is not consistently named across sources. It therefore rests more heavily on the domain expert's operational definition in the annotation guideline than on published taxonomy. Say so. Then note that the geometry analysis in notebook 01 shows Network is nonetheless a distinct, measurable morphology (largest area, lowest elongation, single diffuse region), which is independent support for retaining it.

3. **Validating the rule base is future work with a concrete design.** Attribution could be tested by linking per-egg predictions to flock-level interventions and subsequent crack-rate change, or by correlating predicted cause with contemporaneous shell-thickness and feed-analysis records. State this in 6.6 as a specific next step rather than a generic aspiration.

### Required artefact

Have the hatchery manager sign a one-page attestation: his name, role, years of experience, the statement that he reviewed and approved the taxonomy definitions and the percentage table, and the date. File it in `Administrative/Approvals/` and reproduce it in Appendix G. Without it, the rule base has no provenance in the submitted document.

---

## 6. References (APA 7th)

Aviagen. (2020). *The importance of nutrition on egg shell quality in broiler breeders* (Ross Note). Aviagen Ltd. https://aa-na.aviagen.com/assets/Tech_Center/Ross_Tech_Articles/RossNote-EggShellQuality-2020-EN.pdf

Aviagen. (2020b). *Shell quality in broiler breeders* (Service Bulletin). Aviagen Ltd. https://aa-na.aviagen.com/assets/Tech_Center/AA_Technical_Articles/AA-ServiceBulletin-EggShellQuality-2020-EN.pdf

Aviagen. (2021). *Ross 308 parent stock: Nutrition specifications*. Aviagen Ltd. https://aviagen-na.aviagen.com/assets/Tech_Center/Ross_PS/Ross308-ParentStock-NutritionSpecifications-2021-EN.pdf

Ebeid, T. A., Suzuki, T., & Sugiyama, T. (2014). Effect of hen age, environmental temperature, and oviposition time on egg shell quality and egg shell and serum mineral contents in laying and broiler breeder hens. *Czech Journal of Animal Science*, *59*(9), 406–414. https://doi.org/10.17221/7619-CJAS

Khabisi, M. M., Salahi, A., & Mousavi, S. N. (2012). The influence of egg shell crack types on hatchability and chick quality. *Turkish Journal of Veterinary & Animal Sciences*, *36*(5), 497–506. https://doi.org/10.3906/vet-1103-20

Lee, N.-Y., Lee, M.-J., Kim, Y.-K., Park, J.-C., Park, H.-M., Choi, J.-S., & Kim, J.-S. (2017). Determination of the quality of stripe-marked and cracked eggs during storage. *Asian-Australasian Journal of Animal Sciences*, *30*(7), 1038–1044. https://doi.org/10.5713/ajas.16.0890

Nys, Y., Guyot, N., Bain, M., & Hincke, M. (2023). *Chicken eggshell quality in the late phase of the laying cycle* [Conference presentation]. INRAE. https://hal.inrae.fr/hal-04194886

Perić, L., Mitraković, M., Vekić, M., Đukić-Stojičić, M., Žikić, D., Savić, Đ., Meijerhof, R., & Jotanović, S. (2023). Improving the incubation results by sealing the eggshell of cracked hatching eggs with surgical tape. *Poultry Science*, *102*(3), 102466. https://doi.org/10.1016/j.psj.2022.102466

Priyadarshini, J., Premi, M. S. G., & Arivazhagan, S. (2017). A framework for crack detection of fresh poultry eggs at visible radiation. *Computers and Electronics in Agriculture*, *141*, 142–147. https://doi.org/10.1016/j.compag.2017.07.010

ThePoultrySite. (n.d.). *Optimum egg quality: A practical approach — Shell defects*. https://www.thepoultrysite.com/publications/egg-quality-handbook/14/hairline-cracks

Zhang, Y.-N., Zhang, H.-J., Wang, J., Yue, H.-Y., Qiu, K., Wu, S.-G., & Qi, G.-H. (2020). Transcriptome analysis of the uterus of hens laying eggs differing in cuticle deposition and eggshell quality in the late phase of production. *BMC Genomics*, *21*, 905. https://doi.org/10.1186/s12864-020-07177-7

---

### Verification note

Every reference above was retrieved and checked against its source page. Four details still need confirmation against the full text before submission, because they were taken from abstracts or indexing pages rather than the article body:

* Ebeid et al. (2014) — volume, issue and page range
* Khabisi et al. (2012) — volume, issue and page range
* Priyadarshini et al. (2017) — author list order and page range
* Nys et al. (2023) — whether a journal version exists to cite in place of the HAL record

Also obtain the Gupta (2008) and Farmer et al. (1983) originals if your library has them. Secondary citation is permitted in APA 7 but a direct citation is stronger, and both are load-bearing for the morphology gradient.
