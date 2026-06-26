# Running the app

## Set the environment variables

To run the cloud loop, you must set the relevant environment variable(s) in your terminal before running:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

## Recommended ways to run:

- Slow down the loop frequency (Recommended for Free Tier): Set --hz to 0.2 or 0.25 (1 decision every 4-5 seconds) to stay safely within the 15 RPM limit:

```bash
python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 0.25 --model gemini-2.0-flash
```
- Use GPT-4o-mini (Higher Limits): If you have an OpenAI API key (with even T1 tier, which only requires a $5 deposit), the rate limits are much higher (thousands of RPM) and won't trigger 429 errors at 1 Hz:

```bash
python cloud_ev3_loop.py --host 10.187.118.18 --port 9999 --hz 1.0 --model gpt-4o-mini
```

- Enable Billing in Google AI Studio: If you enable billing on your Google AI Studio account, the Gemini 2.0 Flash limit scales to 2,000 RPM, allowing you to run at --hz 1.0 or higher with Gemini.