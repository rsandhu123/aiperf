# SAP RPT-TS with AIPerf

This integration treats each JSONL row as one complete `POST /ts/predict` request.
AIPerf controls concurrency/rate; request grouping and model options remain part of
the RPT payload.

## Generate a synthetic dataset

```bash
python scripts/generate_rpt_ts_dataset.py \
  --scenario ts1024 \
  --requests 128 \
  --series-per-request 256 \
  --batch-size 128 \
  --output datasets/rpt-ts1024-n256-bs128.jsonl
```

Use `--series-mode identical` to reproduce the existing Locust workload shape,
or leave the default `varied` mode for deterministic but distinct series.

Use `--batch-size 0` to emit `"batch_size": null` and let RPT auto-size.

## Run AIPerf

```bash
export AUTH_TOKEN='...'
export DEPLOYMENT_URL='https://...'

aiperf profile \
  --model sap-rpt-ts \
  --endpoint-type rpt_ts \
  --url "$DEPLOYMENT_URL" \
  --api-key "$AUTH_TOKEN" \
  --header 'AI-Resource-Group: default' \
  --input-file datasets/rpt-ts1024-n256-bs128.jsonl \
  --custom-dataset-type rpt_ts \
  --concurrency 2 \
  --request-count 128
```

If your AIPerf build expects the endpoint URL to exclude the endpoint path, use the
base deployment URL above; the `rpt_ts` plugin registers `/ts/predict` as its path.

## Parameter ownership

- `series_per_request`: generator/benchmark concern; determines `len(data)`.
- `options.batch_size`: RPT model concern; preserved inside each request payload.
- `concurrency` / request rate: AIPerf load-generation concern.

Keeping these independent makes load calculations explicit. For example,
`series_per_request=256`, target length `1024`, concurrency `2` means 524,288
input time points are issued across the two concurrent requests.

## Why request-level JSONL

AIPerf must not regroup individual series. With `cross_learning=true`, series in a
request can influence one another, and hierarchical forecasting aligns
`hierarchical_labels` with `data[]`. Therefore benchmark adapters (synthetic,
FEV, GIFT, TIME) should construct semantically correct complete RPT requests before
AIPerf replays them.
