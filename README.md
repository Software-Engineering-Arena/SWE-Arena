---
title: SE-Arena
emoji: 🛠️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
hf_oauth: true
pinned: false
short_description: The chatbot arena for software engineering
---

# SE Arena: Evaluate Best SE Chatbots with Long-Context Interactions

Welcome to **SE Arena**, an open-source platform designed for evaluating software engineering-focused chatbots. SE Arena benchmarks foundation models (FMs), such as large language models (LLMs), in iterative, context-rich workflows that are characteristic of software engineering (SE) tasks.

## Key Features

- **Advanced Pairwise Comparisons**: Assess chatbots using Elo score, PageRank, and Newman modularity to understand both global performance and task-specific strengths.
- **Interactive Evaluation**: Test chatbots in multi-round conversations tailored for SE tasks like debugging, code generation, and requirement refinement.
- **Open-Source**: Built on [Hugging Face Spaces](https://huggingface.co/spaces/SE-Arena/Software-Engineering-Arena), enabling transparency and fostering community-driven innovation.
- **Transparent Leaderboard**: View real-time model rankings across diverse SE workflows, updated using advanced evaluation metrics.

## Why SE Arena?

Existing evaluation frameworks often do not address the complex, iterative nature of SE tasks. SE Arena fills this gap by:

- Supporting long-context, multi-turn evaluations to capture iterative workflows.
- Allowing anonymous model comparisons to prevent bias.
- Providing rich, multidimensional metrics for more nuanced model evaluations.

## How It Works

1. **Submit a Prompt**: Sign in and input your SE-related task (e.g., debugging, code reviews).
2. **Compare Responses**: Two anonymous chatbots provide responses to your query.
3. **Vote**: Choose the better response, mark as tied, or select "Can't Decide."
4. **Iterative Testing**: Continue the conversation with follow-up prompts to test contextual understanding over multiple rounds.

## Getting Started

### Prerequisites

- A [Hugging Face](https://huggingface.co) account.
- Basic understanding of software engineering workflows.

### Usage

1. Navigate to the [SE Arena platform](https://huggingface.co/spaces/SE-Arena/Software-Engineering-Arena).
2. Sign in with your Hugging Face account.
3. Enter your SE task prompt and start evaluating model responses.
4. Vote on the better response or continue multi-round interactions to test contextual understanding.

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Submit Prompts**: Share your SE-related tasks to enrich our evaluation dataset.
2. **Report Issues**: Found a bug or have a feature request? Open an issue in this repository.
3. **Enhance the Codebase**: Fork the repository, make your changes, and submit a pull request.

## Privacy Policy

Your interactions are anonymized and used solely for improving SE Arena and FM benchmarking. By using SE Arena, you agree to our [Terms of Service](#).

## Future Plans

- **Enhanced Metrics**: Add round-wise analysis and context-aware evaluation metrics.
- **Domain-Specific Sub-Leaderboards**: Rankings focused on tasks like debugging, requirement refinement, etc.
- **Advanced Context Compression**: Techniques like LongRope and SelfExtend to manage long-term memory.
- **Support for Multimodal Models**: Evaluate models that integrate text, code, and other modalities.

## Contact

For inquiries or feedback, please [open an issue](https://github.com/SE-Arena/Software-Engineer-Arena/issues/new) in this repository. We welcome your contributions and suggestions!
