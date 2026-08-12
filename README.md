# test-agent

Agent scaffolded by Boidra Platform's install-agent wizard. It reports OpenTelemetry GenAI traces
to your Boidra Platform instance and deploys to SAP BTP (Cloud Foundry) via GitHub Actions.

## Run locally

```bash
cp .env.example .env         # telemetry endpoint is pre-filled
pip install -r requirements.txt
python agent.py
```

Then open Boidra Platform — a trace for **test-agent** appears under Traces.

## Deploy

Push to `main`. The pipeline builds the Docker image, pushes it to GHCR,
and `cf push`es to BTP. Set the required secrets first (listed at the top of the pipeline file).

## Telemetry

Reports to `http://localhost:5173/otel` as service `test-agent`. The span shape
(invoke_agent → chat → execute_tool) is what the dashboard renders. See `otel_setup.py`.
