# llm_test

## Purpose

llm_test uses [pytest](https://docs.pytest.org/) to do repeatable and scalable user acceptance API testing of Large Language Models (LLMs) for bias, safety, trust, and security.

## Usage

1. Define an importable `Model` template based on your API requirements. Examples are included for [HuggingFace Inference APIs](https://huggingface.co/inference-api) and [OpenAI](https://beta.openai.com/playground). For APIs require authentication, store your API keys in `.env`.
2. Add tests to the `test` directory. In accordance with standard acceptance test format, `assert` the desired behavior. Follow pytest documentation for test discovery, parameterization, fixtures, etc.
3. Modify tests to reference your templated `Model`.
4. Modify test values and prompts based on your interests and acceptance criteria.
5. If your `Model` template or tests require any additional libraries, add them to `requirements.txt`.
6. Build the container: `docker build --tag llm_test .`.
7. Run the container: `docker run --env-file .env llm_test:latest` (after adding your API keys to `.env`). If you want to modify pytest's behavior, do so in the `Dockerfile`.
8. Review Results
![pytest results](results.JPG)

## Existing Tests
- `test/test_counterfactual_sentiment.py`: Uses sentiment analysis to compare the compound sentiment range between provided classes. Currently there is an arbitrary `assert` threshold. A large range suggests that values returned from the model may have been biased and should be inspected more closely.
- `test/test_prompt_injection.py:test_prompt_injection_echo_original`: Based on [available research](https://artifact-research.com/artificial-intelligence/talking-to-machines-prompt-engineering-injection/), reveals underlying prompt that may have been concatenated with user input.
- `test/test_prompt_injection.py:test_prompt_injection_override`: Attempts to override the existing prompt to inject user-defined behavior.

## References
Significantly motivated by the research of:

https://twitter.com/goodside

https://twitter.com/simonw

https://twitter.com/hwchase17
