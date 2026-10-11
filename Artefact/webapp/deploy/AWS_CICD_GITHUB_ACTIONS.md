# AWS CI/CD Deployment with GitHub Actions

End to end: test locally, commit the deployment assets, set up AWS, connect EC2 to GitHub as a
self-hosted runner, and push. After the one-time setup, **every push to `main` that touches the
application rebuilds, tests and redeploys it automatically.**

```
 push to main
     │
     ▼
 ┌──────────────────────────┐   GitHub-hosted runner
 │ 1  Continuous Integration │   syntax check, 28 web tests, artefact contract tests,
 └──────────────┬───────────┘   checks the weights and rule base are committed
                ▼
 ┌──────────────────────────┐   GitHub-hosted runner
 │ 2  Continuous Delivery    │   docker build, push to ECR tagged with the commit SHA
 └──────────────┬───────────┘
                ▼
 ┌──────────────────────────┐   self-hosted runner ON your EC2 instance
 │ 3  Continuous Deployment  │   pull that exact SHA, replace the container, wait for the
 └──────────────────────────┘   models to load, roll back automatically if they do not
```

Allow about two hours for the first run, most of it waiting on AWS and the first image build.

---

## Part A — Test locally first

Do not skip this. Debugging a broken image through a CI log is far slower than through your own
terminal.

### A1. Run the web tests

From the repository root, in your `microcracks` environment:

```powershell
pip install -r Artefact\webapp\requirements.txt
pytest Artefact\webapp\tests -v
pytest Artefact\tests\test_artefact.py -v
```

Expect **28 passed** and **38 passed**.

### A2. Run the application

```powershell
cd Artefact\webapp
.\run_local.ps1
```

Open **http://127.0.0.1:8000** and check, in this order:

1. **http://127.0.0.1:8000/api/v1/ready** shows `"ready": true` and `"models_loaded": 3`.
   If it shows 0, the checkpoints are not in `Artefact\models\final`.
2. **Inspect** — upload E283. Expect *Micro-crack detected — Star*, around 97.9%, with the
   localisation tracing the star.
3. **Batch** — upload the same 20 eggs you used in Streamlit. Expect 5 cracks, 25.0%, 44%
   physiological. **It should match Streamlit exactly** — same model, same preprocessing.
4. Download the CSV and the report and open both.
5. **Model** — check the rule-base table shows nine values.

### A3. Test the Docker image locally

This catches every container problem before AWS is involved. Install Docker Desktop if you have
not, then from the **repository root**:

```powershell
docker build -f Artefact/webapp/Dockerfile -t eggshell-diagnostics:local .
docker run --rm -p 8000:8000 -e SECRET_KEY=local eggshell-diagnostics:local
```

Open http://127.0.0.1:8000/api/v1/ready again. Build from the root, not from `webapp` — the
Dockerfile copies `Artefact/src`, the checkpoints and the rule base as well.

Expect about **1.2 GB**. Check with `docker images`. If it is near 6 GB, the CUDA build of PyTorch
got in and the `--index-url` line in the Dockerfile needs checking.

---

## Part B — Commit the deployment assets

GitHub's build runner only sees what is committed, so the three final checkpoints and the rule
base must be in the repository. Your repository is private, which is what makes this acceptable.

### B1. Update `.gitignore`

Paste the contents of `gitignore_deployment_block.txt` at the **end** of the root `.gitignore`.
It un-ignores exactly three `.pt` files and one CSV; the Stage A and B checkpoints and the 400 MB
cache stay out.

### B2. Add and verify

```powershell
git add .gitignore
git add Artefact/models/final/final_efficientnet_b0_multitask_seg_seed*.pt
git add Artefact/data/processed/root_cause_rules_recovered.csv
git add Artefact/webapp Artefact/src Artefact/tests .github
git status
```

`git status` should list **three** `.pt` files, about 15 MB each. If you see dozens, the
`.gitignore` block is not at the end of the file, or a Stage A checkpoint slipped through — stop
and fix it before committing, because removing large files from git history afterwards is painful.

**Do not push yet.** The pipeline will fail without the AWS setup.

---

## Part C — AWS setup

### C1. Choose a region

**`ap-south-1` (Mumbai)** is closest to Sri Lanka. Use the same region for everything below. A
repository in one region and an instance in another is the most common reason a first deployment
fails to pull its image.

### C2. Create an IAM user for GitHub Actions

IAM → Users → **Create user** → name `github-actions-eggshell` → do **not** give console access.

Attach policies directly:

| Policy | Why |
|---|---|
| `AmazonEC2ContainerRegistryFullAccess` | push from the build runner, pull on the instance |

`AmazonEC2FullAccess`, which the example template includes, is **not needed** by this pipeline:
the workflow never creates or modifies instances, and you are launching the instance yourself in
the console. Leaving it off means a leaked key cannot launch machines on your account.

Then: the user → **Security credentials** → **Create access key** → *Application running outside
AWS* → copy the **Access key ID** and **Secret access key** now. The secret is shown once.

### C3. Create the ECR repository

ECR → **Create repository** → Private → name **`eggshell-diagnostics`** → create.

Copy the URI. It looks like:

```
123456789012.dkr.ecr.ap-south-1.amazonaws.com/eggshell-diagnostics
```

The part before the slash, `123456789012.dkr.ecr.ap-south-1.amazonaws.com`, is the registry.

Optional but sensible: **Lifecycle policy** → keep only the last 5 images. Each is 1.2 GB and ECR
charges for storage.

### C4. Launch the EC2 instance

EC2 → **Launch instance**.

| Setting | Value | Why |
|---|---|---|
| Name | `eggshell-diagnostics` | |
| AMI | **Ubuntu Server 24.04 LTS** | |
| Instance type | **t3.medium** (2 vCPU, 4 GB) | the ensemble needs ~1.5 GB; **t2.micro and t3.small run out of memory** |
| Key pair | create one, download the `.pem` | |
| Storage | **30 GB gp3** | the default 8 GB fills after one image pull plus Docker's working space |
| Security group | see below | |

Security group inbound rules:

| Type | Port | Source |
|---|---|---|
| SSH | 22 | **My IP** only |
| HTTP | 80 | Anywhere (0.0.0.0/0) |

Launch it.

**Allocate an Elastic IP** (EC2 → Elastic IPs → Allocate → Associate with this instance).
Without one, the public address changes every time the instance is stopped, and the URL in your
dissertation stops working.

### C5. Install Docker on the instance

Connect: EC2 → the instance → **Connect** → EC2 Instance Connect. Then:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

docker run --rm hello-world          # confirms docker works without sudo
```

The workflow uses the AWS GitHub Actions to log in to ECR, so the AWS CLI is not needed on the
instance.

---

## Part D — Connect EC2 to GitHub as a self-hosted runner

### D1. Register the runner

On GitHub: your repository → **Settings → Actions → Runners → New self-hosted runner** →
**Linux**, **x64**.

GitHub shows a list of commands. Run them **one at a time** in the EC2 terminal. They download the
runner, then configure it with a token that expires in an hour.

When `./config.sh` asks:

| Prompt | Answer |
|---|---|
| runner group | press Enter |
| runner name | `eggshell-ec2` |
| additional labels | press Enter |
| work folder | press Enter |

### D2. Install it as a service — do not skip this

The template ends with `./run.sh`. **That only runs while your terminal is open.** Close the
browser tab and the runner disappears, and the next deployment waits forever on "Waiting for a
runner to pick up this job".

Install it as a system service instead, from the runner directory:

```bash
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

Back on GitHub → Settings → Actions → Runners, it should show **Idle** with a green dot.

---

## Part E — GitHub secrets

Repository → **Settings → Secrets and variables → Actions → New repository secret**.

| Secret | Example value |
|---|---|
| `AWS_ACCESS_KEY_ID` | from step C2 |
| `AWS_SECRET_ACCESS_KEY` | from step C2 |
| `AWS_REGION` | `ap-south-1` |
| `ECR_REPOSITORY_NAME` | `eggshell-diagnostics` |
| `FLASK_SECRET_KEY` | any long random string, see below |

Generate the Flask secret in PowerShell:

```powershell
-join ((48..57) + (97..102) | Get-Random -Count 64 | % {[char]$_})
```

`AWS_ECR_LOGIN_URI`, which the example template uses, is **not needed** here: the workflow reads
the registry address from the ECR login step, which removes one place to make a typo.

---

## Part F — Deploy

### F1. Push

```powershell
git commit -m "feat(webapp): Flask application with CI/CD to AWS"
git push origin main
```

### F2. Watch it

Repository → **Actions** → the running workflow. Three jobs in sequence:

| Job | Takes | Fails most often because |
|---|---|---|
| Continuous Integration | about 1 min | a checkpoint or the rule CSV was not committed |
| Continuous Delivery | 6 to 10 min the first time | wrong `AWS_REGION` or repository name |
| Continuous Deployment | 1 to 3 min | the runner is offline (step D2), or the instance is too small |

The deployment job will not report success until `/api/v1/ready` confirms the models loaded. If
they do not load within 200 seconds it prints the container log and **rolls back to the previous
image automatically**.

### F3. Open it

```
http://<your-elastic-ip>/
http://<your-elastic-ip>/api/v1/ready
```

Repeat the checks from step A2 against the live URL. The results should match your local run
exactly.

---

## Part G — After deploying

- [ ] Put the URL in the dissertation. The module specification requires a link to the
      implementation.
- [ ] Record the URL, the commit SHA and the date in `Administrative/Progress/`.
- [ ] **Add HTTPS before showing it to anyone outside the project** — see below.
- [ ] Stop the instance between demonstrations.

### HTTPS

Plain HTTP is fine for testing. It is not fine in front of farm staff, because your participant
information sheet promises protected handling of their data. Two routes:

- **Simplest:** a free domain or subdomain pointed at the Elastic IP, then Caddy as a reverse
  proxy on the instance, which obtains a Let's Encrypt certificate automatically.
- **AWS-native:** an Application Load Balancer with an ACM certificate in front of the instance.
  More setup, and the load balancer costs about 18 USD a month on its own.

### Cost

| Item | Approximate cost |
|---|---|
| t3.medium running | 0.045 USD per hour, about 33 USD a month continuously |
| 30 GB gp3 storage | about 2.5 USD a month, charged even while stopped |
| Elastic IP | free while attached to a running instance, small hourly charge while stopped |
| ECR, 5 images | a few cents a month |

**Stop the instance when you are not demonstrating.** A stopped t3.medium costs nothing but its
disk. The self-hosted runner reconnects by itself when the instance starts again, because it was
installed as a service.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Deploy job stuck on "Waiting for a runner" | runner not running as a service | Part D2 |
| `no basic auth credentials` on pull | region mismatch between ECR and secrets | make `AWS_REGION` match the repository's region |
| Container exits, log shows `Killed` | out of memory | t3.medium or larger |
| `no space left on device` | 8 GB root volume | resize the volume to 30 GB, or `docker system prune -af` |
| `/api/v1/ready` shows `models_loaded: 0` | checkpoints not in the image | check they are committed, Part B2 |
| Page loads but shows *Service degraded* | rule base CSV missing from the image | Part B2 |
| CI fails at *Deployment assets are committed* | `.gitignore` order wrong | the block must be at the **end** of the file |
| Browser cannot reach the site at all | port 80 not open | security group, Part C4 |
| URL stopped working after a restart | no Elastic IP | Part C4 |
| Batch says "expired" straight after analysing | more than one worker or instance | keep `GUNICORN_WORKERS=1`, one instance |

---

## Security notes

- **Rotate the access key** if it is ever pasted anywhere other than GitHub Secrets.
- **The application has no login.** Anyone with the URL can use it. Acceptable for a
  demonstration; before real farm data goes in, add an IP allow-list in the security group or put
  it behind Amazon Cognito.
- **No uploaded image is stored.** Images are decoded in memory and discarded after the response,
  which is consistent with the data-handling statement in your participant information sheet.
- A cleaner long-term design gives the instance an **IAM role** with read-only ECR access, so the
  deployment job needs no keys on the instance at all. The pipeline as written uses the same keys
  in both places for simplicity.
