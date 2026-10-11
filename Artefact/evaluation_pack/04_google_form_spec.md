# Google Form — build specification

Reproduce this exactly. **The question titles are the column headers in the CSV export**, and
notebook 12 matches them on the `Q1.` / `O1.` / `D1.` prefixes. Keep the prefixes and the numbering
or the analysis will not find your data.

**Form settings**
- Title: *Eggshell Micro-Crack Prototype — Practitioner Feedback*
- Collect email addresses: **off**
- Limit to one response: off
- Responses tab, link to Google Sheets, then File, Download, CSV

---

## Section 1 — Introduction (description only, no questions)

> Thank you for taking the time to look at this prototype.
>
> This is part of a BSc research project at NIBM / Coventry University. You have just seen a
> demonstration of a system that detects micro-cracks in candled hatching eggs, classifies the type
> of crack, and suggests whether the likely cause is shell quality or handling.
>
> I am asking for your professional opinion on whether this kind of information would be useful in
> your work. **There are no right answers, and critical answers are more useful to me than polite
> ones.**
>
> Your name is not collected. Responses are reported anonymously in a student dissertation.
> Participation is voluntary and you may stop at any point.
>
> It takes about eight minutes.

---

## Section 2 — About you

Short answer / multiple choice as marked. Mark all **Required**.

**D1. What is your role?** *(multiple choice)*
- Hatchery manager
- Production manager
- Farm manager
- Supervisor
- Technician
- Other

**D2. How many years have you worked in poultry production?** *(multiple choice)*
- Less than 2
- 2 to 5
- 6 to 10
- 11 to 20
- More than 20

**D3. Approximately how many birds are on the farm you work with?** *(multiple choice)*
- Under 10,000
- 10,000 to 50,000
- 50,000 to 100,000
- Over 100,000
- Prefer not to say

**D4. How are cracked eggs currently identified on your farm?** *(checkboxes, multiple allowed)*
- Visual inspection by hand
- Candling
- Tapping or sound
- An automated machine
- They are usually not identified before setting
- Other

**D5. How often are cracked hatching eggs a problem for you?** *(multiple choice)*
- Never
- Occasionally
- Regularly
- Constantly

---

## Section 3 — Would this be useful?

Add this description to the section:

> For each statement, choose the option closest to your view. 1 means strongly disagree, 5 means
> strongly agree.

All of the following are **Linear scale, 1 to 5**, label 1 as `Strongly disagree` and 5 as
`Strongly agree`. Mark all **Required**.

- **Q1.** This system would help us find micro-cracks that are currently being missed.
- **Q2.** Knowing the crack type (hairline, star or network) would be useful in my work.
- **Q3.** Knowing whether the likely cause is shell quality or handling would help me decide what to fix.
- **Q4.** A system like this could reduce hatching-egg losses on our farm.
- **Q5.** The batch report would be useful when discussing problems with my team or with management.

---

## Section 4 — Would it fit how you work?

- **Q6.** The information the system shows is easy to understand.
- **Q7.** Our staff could learn to use something like this without much training.
- **Q8.** Photographing each egg over a candling light would fit into our current egg-handling routine.
- **Q9.** The amount of information on screen is about right, not too much and not too little.

---

## Section 5 — Would you trust it?

- **Q10.** I would trust the system when it says an egg is cracked.
- **Q11.** The causes it suggests match what I would conclude from my own experience.
- **Q12.** The actions it recommends are things we could realistically carry out.

---

## Section 6 — Would you use it?

- **Q13.** If this were available, I would want to try it on our farm.
- **Q14.** I would prefer this to how we check eggs now.
- **Q15.** A future automated version, with a camera on the egg belt and no manual photography, would be worth investing in.

---

## Section 7 — In your own words

All **Paragraph** type. Mark **O1** and **O3** required; the rest optional.

- **O1.** What is the biggest cause of hatching-egg loss on your farm at the moment?
- **O2.** Where in your process would a system like this fit best?
- **O3.** What would stop you from using it?
- **O4.** What would you add or change?
- **O5.** Was there anything shown today that you did not believe, or were not sure about?

---

## Section 8 — Closing (description only)

> Thank you. If you would like a summary of the findings, email `[YOUR EMAIL]` and I will send one
> when the project is complete.

---

## After collecting

1. Responses tab, green Sheets icon, open in Google Sheets
2. File, Download, Comma-separated values
3. Save as `Artefact/evaluation_pack/responses.csv`
4. Run `notebooks/12_stakeholder_appraisal.ipynb`

Do not rename the columns. The notebook reads the `Q`, `O` and `D` prefixes directly.
