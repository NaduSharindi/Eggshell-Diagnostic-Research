# Stakeholder Appraisal Pack

Objective 4: *evaluate the prototype with poultry production personnel.*

**This is not a usability study.** Participants watch a demonstration and give a professional
judgement; they do not operate the system and complete no timed tasks. The instrument is therefore
built on the Technology Acceptance Model (Davis, 1989) with an added trust construct, not on the
System Usability Scale. Administering SUS after a demonstration produces a number that looks like
a usability score and is not one.

Report it as an **expert appraisal of perceived usefulness and adoption intention**. Nothing about
usability, learnability, task time or error rates may be claimed, because none of it was measured.

This pack contains everything needed to run the evaluation and nothing that presumes its outcome.
No results are fabricated anywhere; the analysis notebook reads whatever responses you collect and
reports them, including a small or incomplete sample.

| File | Purpose | Used when |
|---|---|---|
| `01_participant_information_sheet.md` | What the study is, what participation involves | Sent or read before the session |
| `02_consent_form.md` | Written consent, signed before anything is shown | Start of session |
| `03_demonstration_script.md` | What to show, in what order, and what to say | During the session |
| `04_google_form_spec.md` | Exact questions to build the Google Form from | Once, before any session |
| `responses_template.csv` | The column layout the form export produces | Reference only |
| `responses.csv` | Your downloaded Google Forms export | After the sessions |
| `../notebooks/12_stakeholder_appraisal.ipynb` | Scores the four constructs, exports open responses | After all sessions |

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
| 3 min | Context: what the system is for, and plainly what it is not |
| 14 min | Demonstration: one egg, then a tray, then the automated future |
| 8 min | Google Form, completed by the participant alone |
| 3 min | Anything they want to add, recorded in your notes |

**Show every participant the same things in the same order.** If the demonstration varies, the
responses are not comparable and cannot be pooled.

**Leave them alone with the form.** Standing over someone while they rate your work inflates every
number.
