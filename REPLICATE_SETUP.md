# Using Replicate API for Faster Processing

The clothing detection app now supports cloud processing via Replicate API for faster results (~5-10s vs 60s+ locally).

## Setup

1. **Get your Replicate API token**:
   - Go to https://replicate.com/account/api-tokens
   - Sign up/login and copy your API token

2. **Set the environment variable**:
   ```bash
   export REPLICATE_API_TOKEN="your_token_here"
   ```

3. **Run the app**:
   ```bash
   ./run.sh
   ```

4. **Select "Replicate API" from the model dropdown** in the web interface

## Benefits

- **Speed**: 5-10 seconds vs 60+ seconds for local processing
- **No GPU required**: Runs on Replicate's cloud infrastructure
- **Same quality**: Uses the same SAM2 model with optimized parameters

## Costs

Replicate charges per prediction. Check current pricing at https://replicate.com/pricing

## Troubleshooting

If you see "Replicate API failed", check:
1. Your API token is set correctly
2. You have credits in your Replicate account
3. Your internet connection is working