# User Acceptance Testing — Facilitator Script

**Participant code** P___    **Date** ______    **Facilitator** ______

Read the parts in quotation marks aloud. Everything else is instruction to you.

## Setup checklist

- [ ] App running, **Inspect** tab open, no image loaded
- [ ] Age class set to **Peak production**, flock and operator fields blank
- [ ] Test images ready in two folders: `single/` (5 eggs) and `tray/` (20 eggs)
- [ ] Ground truth for those eggs noted on your sheet, **not visible to the participant**
- [ ] Stopwatch, notes template, consent form signed

**Do not demonstrate the tool first.** Measuring usability after a walkthrough measures your
walkthrough.

## Framing, about two minutes

> "This tool looks at a photograph of a candled egg and tells you whether it has a micro-crack,
> what type of crack it is, and whether the likely cause is shell quality or handling. I am going
> to ask you to do six short things with it. I will not help unless you are properly stuck.
>
> Please say out loud what you are thinking as you go, including when something is confusing or
> annoying. If you get stuck, that is useful information about the tool, not about you.
>
> There is no right answer and nothing is being judged except the software."

## Scoring each task

| Code | Meaning |
|---|---|
| **U** | Completed unaided |
| **P** | Completed after a prompt from the facilitator (record the prompt) |
| **F** | Not completed, or completed incorrectly |

Record time from the end of your instruction to the participant's stated answer. Note every
hesitation, wrong click and spontaneous comment. **Prompt only after 60 seconds of visible
difficulty**, and record that you did.

---

## Task 1 — Inspect a single egg

> "Here is a photograph of one egg from a peak-production flock. Use the tool to tell me whether
> this egg has a micro-crack."

Use a **clearly cracked** egg. Target under 60 s.

Outcome: U / P / F    Time: ____ s    Answer correct: Y / N

Notes:

---

## Task 2 — Find the crack type and the likely cause

> "Still with that egg. What type of crack is it, and what does the tool say is the most likely
> cause?"

Looking for: the morphology name, and that they read the physiological/mechanical split rather
than only the interpretation text.

Outcome: U / P / F    Time: ____ s    Answer correct: Y / N

Did they notice the percentage split unprompted?  Y / N

Notes:

---

## Task 3 — Spot an uncertain result

> "Here are four more eggs, one at a time. For each, tell me whether you would act on the tool's
> answer or check the egg yourself first."

Include **one egg the tool flags for manual review**. The task is whether the flag is noticed and
understood without explanation.

Outcome: U / P / F    Time: ____ s

Did they notice the low-confidence flag?  Y / N
Did they correctly say they would check that one by hand?  Y / N

Notes:

---

## Task 4 — Analyse a tray

> "This folder has twenty eggs from one flock. Use the tool to tell me how many have micro-cracks
> and what the main cause looks like across the whole tray."

Tests whether the Batch tab is discoverable without being pointed at. Target under 180 s.

Outcome: U / P / F    Time: ____ s

Did they find the Batch tab unaided?  Y / N
Crack count correct: Y / N    Dominant cause read correctly: Y / N

Notes:

---

## Task 5 — Decide what to do

> "Based on what the tool just told you about that tray, what would you actually do tomorrow
> morning?"

The important one. A tool that reports correctly but prompts no action has failed at its purpose.
Record the answer verbatim.

Outcome: U / P / F    Time: ____ s
Action named is specific and plausible: Y / N
Did they reference the suggested actions, or their own judgement, or both?

Verbatim answer:

---

## Task 6 — Find the limitations

> "Suppose you were deciding whether to trust this tool on your own farm. Can you find out what it
> is not good at?"

Tests whether the Model tab and the limitations are discoverable. Honesty is only useful if it can
be found.

Outcome: U / P / F    Time: ____ s

Did they find the Model tab?  Y / N
Did they identify at least one real limitation?  Y / N    Which:

Notes:

---

## Debrief, about ten minutes

Ask all three and record the answers close to verbatim.

1. **"If this were installed at your hatchery tomorrow, would you use it? What would stop you?"**

2. **"Was there anything the tool told you that you did not believe? What would you need to see
   to believe it?"**

3. **"What is the single thing you would change first?"**

Then hand over the SUS questionnaire and **leave them to complete it alone**. Standing over a
participant while they rate your software inflates the score.

## Facilitator reflection, immediately after

- Anything that confused more than one participant so far:
- Anything the tool did that was wrong, and how they reacted:
- Your own one-line impression of the session:
