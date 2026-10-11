# Deploying to AWS

Three options, cheapest first. **App Runner is the one to use** unless you have a reason not to:
it takes a container image and gives back an HTTPS URL, with no load balancer, no VPC and no
instance to patch.

Before any of them, confirm the application runs locally.

---

## 0. Run it locally first

```bash
cd Artefact/webapp
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python wsgi.py
```

Open `http://127.0.0.1:8000`. Check:

- `/api/v1/ready` returns `"ready": true` and `"models_loaded": 3`
- Inspect correctly classifies an egg whose answer you know
- Batch produces a report you can read

If `/api/v1/ready` reports `models_loaded: 0`, the checkpoints are not where the application
expects. Set `EGG_ARTEFACT_ROOT` to the absolute path of your `Artefact` directory.

---

## 1. Build the image

Build from the **repository root**, not from `webapp`, because the Dockerfile copies `Artefact/src`
and `Artefact/models/final` as well.

```bash
docker build -f Artefact/webapp/Dockerfile -t eggshell-diagnostics:1.0 .
docker run --rm -p 8000:8000 eggshell-diagnostics:1.0
```

Expect roughly 1.2 GB. If it comes out near 6 GB, the CUDA build of torch was installed — check the
`--index-url` line in the Dockerfile survived.

The weights ship **inside** the image. For a student deployment that is the simplest thing that
works. For anything beyond it, put them in S3 and download at start-up so the image is not rebuilt
for every retrain.

---

## 2. Push to ECR

```bash
export AWS_REGION=ap-south-1          # Mumbai is the closest region to Sri Lanka
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/eggshell-diagnostics

aws ecr create-repository --repository-name eggshell-diagnostics --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag eggshell-diagnostics:1.0 $REPO:1.0
docker push $REPO:1.0
```

---

## 3. Option A — App Runner (recommended)

Console: **App Runner → Create service → Container registry → Amazon ECR**, choose the image.

| Setting | Value | Why |
|---|---|---|
| Port | 8000 | |
| CPU / memory | 2 vCPU, 4 GB | the ensemble needs roughly 1.5 GB resident; 2 GB will OOM under load |
| Health check path | `/api/v1/health` | liveness, not readiness, or a missing checkpoint cycles the service forever |
| Health check interval | 20 s, timeout 5 s, unhealthy threshold 5 | first start-up loads the models |
| Environment | `TORCH_NUM_THREADS=2`, `GUNICORN_THREADS=4`, `SECRET_KEY=<random>` | |
| Auto scaling | min 1, max 1 | **important — see below** |

**Keep max concurrency at one instance.** Batch results live in an in-process store keyed by job
id. With two instances behind a load balancer, the redirect after analysis can land on the instance
that has never seen that job and the user is told their batch expired seconds after creating it.

To scale past one instance, move `app/jobs.py` to ElastiCache Redis first. Until then, one instance
is a correctness requirement, not a cost saving.

App Runner gives you an HTTPS URL immediately. Set `SECRET_KEY` to something random and keep it out
of the repository.

---

## 4. Option B — EC2 with Docker

Cheapest if you already have an instance, and the only option if you need to SSH in and look at
things.

- **t3.medium** (2 vCPU, 4 GB). t3.small will be killed loading the ensemble.
- Security group: 80 and 443 from anywhere, 22 from your IP only.

```bash
sudo yum install -y docker && sudo systemctl enable --now docker
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS \
  --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
sudo docker run -d --restart unless-stopped -p 80:8000 \
  -e SECRET_KEY=$(openssl rand -hex 32) -e TORCH_NUM_THREADS=2 \
  --name eggshell $REPO:1.0
```

For HTTPS put CloudFront or an ALB with an ACM certificate in front. **Do not run it on plain HTTP
in front of farm staff** — the consent and data-handling claims in your ethics documentation assume
the connection is protected.

---

## 5. Option C — ECS Fargate

Use this only if the deployment has to sit inside a VPC alongside other services. It needs a task
definition, a cluster, an ALB, target groups and security groups. For a single-container
undergraduate artefact it is more moving parts than the problem justifies, and every one of them
is something to debug the week before submission.

If you do: ALB target group health check `/api/v1/health`, deregistration delay 60 s, task 2 vCPU /
4 GB, desired count **1** for the same reason as above.

---

## 6. After deploying

- [ ] `GET /api/v1/ready` returns `"ready": true`
- [ ] Inspect an egg whose answer you know and confirm it matches the local result
- [ ] Run a tray and download the report
- [ ] Confirm the URL is HTTPS before showing it to anyone outside the project
- [ ] Record the URL, the image tag and the date in `Administrative/Progress/`
- [ ] Put the URL in the dissertation as the implementation link required by the module spec

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | random per start | set it, or sessions break on restart |
| `PORT` | 8000 | |
| `EGG_ARTEFACT_ROOT` | `/srv/artefact` in the image | |
| `EGG_THRESHOLD` | 0.30 | the evidenced default; a change is logged as a warning |
| `EGG_LOW_CONFIDENCE` | 0.60 | |
| `EGG_MAX_BATCH` | 240 | images per batch |
| `EGG_JOB_TTL` | 3600 | seconds a batch result is retained |
| `MAX_UPLOAD_MB` | 400 | per request |
| `TORCH_NUM_THREADS` | 2 | match the vCPU count |
| `GUNICORN_THREADS` | 4 | |
| `GUNICORN_WORKERS` | 1 | **leave at 1**, see section 3 |
| `EGG_LAZY_LOAD` | 0 | 1 defers model loading to the first request |

---

## Cost, at the time of writing

App Runner 2 vCPU / 4 GB is roughly 0.08 USD per hour while running, about 55 USD a month if left
on continuously. **Pause the service between demonstrations.** For a student project running it
only when needed costs a few dollars.

ECR storage for a 1.2 GB image is a few cents a month.

---

## Known operational limits

- **One instance only** until the job store moves to Redis.
- **Weights are baked into the image**, so retraining means rebuilding and redeploying.
- **CPU inference** at roughly 1.3 s per egg; a 240-image tray takes about five minutes.
- **No authentication.** Anyone with the URL can use it. For a demonstration that is acceptable;
  before any real farm data goes in, put it behind Cognito or an IP allow-list.
- **No persistence.** Nothing is written to disk and no uploaded image is retained, which is
  deliberate and consistent with the data-handling statement in the participant information sheet.
