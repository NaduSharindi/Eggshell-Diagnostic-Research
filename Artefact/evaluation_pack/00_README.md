# Usability and User Acceptance Evaluation Pack

Objective 4: *evaluate the Streamlit decision-support dashboard with poultry production managers.*

This pack contains everything needed to run the evaluation and nothing that presumes its outcome.
No results are fabricated anywhere; the analysis notebook reads whatever responses you collect and
reports them, including a small or incomplete sample.

| File | Purpose | Used when |
|---|---|---|
| `01_participant_information_sheet.md` | What the study is, what participation involves | Sent before the session |
| `02_consent_form.md` | Written consent, signed before anything is shown | Start of session |
| `03_uat_script.md` | Facilitator script and six timed tasks | During the session |
| `04_sus_questionnaire.md` | System Usability Scale plus four domain items | End of session |
| `05_facilitator_notes_template.md` | One per participant, for observations and quotes | During and after |
| `responses_template.csv` | The exact structure notebook 12 reads | Data entry after sessions |
| `../notebooks/12_usability_analysis.ipynb` | Scores SUS, task success, times, qualitative themes | After all sessions |

## Before you run anything

**Align the wording with your approved ethics documentation.** Sections marked
`[CHECK AGAINST ETHICS APPROVAL]` in the information sheet and consent form are the ones most
likely to diverge from what your institution approved. The approved wording wins; edit these files
to match it, do not edit your approval to match these files.

**Sample size.** The protocol is written for ten participants. The analysis notebook reports
whatever it is given and states the sample size prominently. Five participants honestly reported
is worth more than ten claimed and not delivered. If you get fewer than eight, describe the study
as indicative rather than inferential, and say so in Section 5.8.

## Running order for one session, about 45 minutes

| | |
|---|---|
| 5 min | Information sheet, questions, consent signed |
| 5 min | Context: what the system is for, what it does not do. Do **not** demonstrate it |
| 20 min | Six tasks from the UAT script, think-aloud, timed |
| 5 min | SUS and the four domain items, completed by the participant alone |
| 10 min | Debrief: three open questions, recorded as notes or audio if consented |

**Do not demonstrate the system before the tasks.** A usability score measured after a walkthrough
measures your explanation, not the interface.
