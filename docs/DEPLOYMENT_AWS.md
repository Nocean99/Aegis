# Deploying the Analyst Dashboard on AWS (Free Tier)

End-to-end: container → ECR → a free-tier EC2 instance → HTTPS with real
auth, health checks, and metrics. Total cost on a new AWS account: $0 for
12 months (t2.micro/t3.micro 750 hrs/month free); roughly $8–10/month after.

Architecture (deliberately boring):

```text
Browser ──HTTPS──► Caddy (TLS) ──► dashboard container (:8010)
                                    ├── /healthz   (public, for checks)
                                    ├── /metrics   (auth-protected, Prometheus text)
                                    └── /          (session auth)
                       EC2 t3.micro, Docker, restart: unless-stopped
```

## 0. Local sanity check first

```bash
# Generate the credential (do this once, save the output):
python3 -m autonomy.dashboard_auth hash 'choose-a-strong-password'

# Put it in .env (gitignored):
cat > .env <<'EOF'
ANALYST_USERNAME=analyst
ANALYST_PASSWORD_HASH=pbkdf2_sha256$310000$...paste-the-hash...
EOF

docker compose up --build analyst-dashboard
# Verify: http://localhost:8010 redirects to /login; log in; check /healthz.
```

## 1. Push the image to ECR

```bash
AWS_REGION=ca-central-1            # Halifax-adjacent; pick yours
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=aegis-analyst-dashboard

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO .
docker tag $REPO:latest $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

## 2. Launch the instance

Console → EC2 → Launch instance:

- AMI: Amazon Linux 2023, Architecture: x86_64
- Instance type: t3.micro (or t2.micro where t3 isn't free-tier)
- Key pair: create one, keep the .pem safe
- Security group: inbound 22 (SSH, your IP only), 80, 443. Do NOT open 8010
  to the world — the container listens only on the instance's localhost and
  Caddy fronts it.
- Storage: 8 GB gp3 (free tier covers 30)

Then on the instance:

```bash
ssh -i key.pem ec2-user@<public-ip>
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user   # then reconnect

# Pull the image (attach an IAM role with AmazonEC2ContainerRegistryReadOnly,
# or use aws configure with a limited user):
aws ecr get-login-password --region ca-central-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ca-central-1.amazonaws.com
docker pull <account>.dkr.ecr.ca-central-1.amazonaws.com/aegis-analyst-dashboard:latest
```

## 3. Run it with auth + TLS

```bash
mkdir -p ~/aegis/logs && cd ~/aegis

cat > .env <<'EOF'
ANALYST_USERNAME=analyst
ANALYST_PASSWORD_HASH=pbkdf2_sha256$310000$...your-hash...
ANALYST_COOKIE_SECURE=1
EOF

# Dashboard, bound to localhost only — Caddy is the public face:
docker run -d --name aegis-dashboard --restart unless-stopped \
  --env-file .env -p 127.0.0.1:8010:8010 \
  -v ~/aegis/logs:/app/logs \
  <account>.dkr.ecr.ca-central-1.amazonaws.com/aegis-analyst-dashboard:latest

# Caddy: automatic HTTPS with a free DuckDNS (or any) domain.
# Point e.g. aegis-demo.duckdns.org at the instance's public IP first.
cat > Caddyfile <<'EOF'
aegis-demo.duckdns.org {
    reverse_proxy 127.0.0.1:8010
}
EOF
docker run -d --name caddy --restart unless-stopped --network host \
  -v ~/aegis/Caddyfile:/etc/caddy/Caddyfile \
  -v caddy_data:/data caddy:2
```

Visit https://aegis-demo.duckdns.org → login page → dashboard. Done.

(No domain? Skip Caddy, open 8010 in the security group restricted to your
own IP, browse http://\<ip\>:8010, and leave ANALYST_COOKIE_SECURE=0. Fine
for a personal demo; use the TLS path for anything you share.)

## 4. Monitoring in operation

- `https://<host>/healthz` — public JSON: `{"ok": true, "uptime_s": ...,
  "requests_total": ..., "auth_failures_total": ...}`. Wire it to a free
  UptimeRobot monitor for downtime email alerts.
- `https://<host>/metrics` — Prometheus text format (login required):
  request counts by route/status, latency sums, failed logins, active
  sessions. Scrape it with Prometheus/Grafana Cloud free tier, or just curl
  it with the session cookie.
- Structured logs — every request is one JSON line on stdout:

  ```bash
  docker logs aegis-dashboard --since 1h | python3 -m json.tool --json-lines
  ```

  Ship them with `aws logs` via the awslogs Docker log driver if you want
  CloudWatch retention (free tier: 5 GB).

## 5. Updating a deployment

```bash
docker build -t $REPO . && docker tag ... && docker push ...   # local
docker pull ... && docker stop aegis-dashboard && docker rm aegis-dashboard \
  && docker run ...                                            # instance, same flags
```

Reports live in `~/aegis/logs` on the host volume, so redeploys keep all
mission history and analyst reviews.

## Security posture summary

Salted PBKDF2-SHA256 credentials from env (never in the image or repo),
HttpOnly/SameSite=Strict session cookies (Secure under TLS), per-IP login
rate limiting, non-root container user, path-traversal-hardened file
serving restricted to allowed roots, container port never exposed publicly,
TLS terminated by Caddy with auto-renewed certificates.

See `docs/LOAD_HANDLING.md` for capacity reasoning and the scaling path.
