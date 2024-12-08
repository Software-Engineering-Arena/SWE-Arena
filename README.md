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

# SE Arena: Explore and Test the Best SE Chatbots with Long-Context Interactions

Welcome to **SE Arena**, an open-source platform for evaluating software engineering-focused chatbots. SE Arena is designed to benchmark foundation models (FMs), including large language models (LLMs), in iterative and context-rich workflows characteristic of software engineering (SE) tasks.

## Key Features

- **Interactive Evaluation**: Test chatbots in multi-round conversations tailored for debugging, code generation, and requirement refinement.
- **Transparent Leaderboard**: View model rankings across diverse SE workflows, updated in real-time using advanced metrics.
- **Advanced Pairwise Comparisons**: Evaluate chatbots using metrics like Elo score, PageRank, and Newman modularity to understand their global dominance and task-specific strengths.
- **Open-Source**: Built on [Hugging Face Spaces](https://huggingface.co/spaces/SE-Arena/Software-Engineering-Arena), fostering transparency and community-driven innovation.

## Why SE Arena?

Existing evaluation frameworks often fall short in addressing the complex, iterative nature of SE tasks. SE Arena fills this gap by:

- Supporting long-context, multi-turn evaluations.
- Allowing comparisons of anonymous models without bias.
- Providing rich, multidimensional metrics for nuanced evaluations.

## How It Works

1. **Submit a Prompt**: Sign in and input your SE-related task (e.g., debugging, code reviews).
2. **Compare Responses**: Two chatbots respond to your query side-by-side.
3. **Vote**: Choose the better response, mark as tied, or select "Can't Decide."
4. **Iterative Testing**: Continue the conversation with follow-up prompts to test long-context understanding.

## Metrics Used

SE Arena goes beyond traditional Elo scores by incorporating:

- **Eigenvector Centrality**: Highlights models that perform well against high-quality competitors.
- **PageRank**: Accounts for cyclic dependencies and emphasizes importance in dense sub-networks.
- **Newman Modularity**: Groups models into clusters based on similar performance patterns, helping users identify task-specific expertise.

## Getting Started

### Prerequisites

- A [Hugging Face](https://huggingface.co) account.
- Basic knowledge of software engineering workflows.

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

Your interactions are anonymized and used solely for improving SE Arena and foundation model benchmarking. By using SE Arena, you agree to our [Terms of Service](#).

## Future Plans

- **Enhanced Metrics**: Add round-wise analysis and context-aware metrics.
- **Domain-Specific Sub-Leaderboards**: Focused rankings for debugging, requirement refinement, etc.
- **Integration of Advanced Context Compression**: Techniques like LongRope and SelfExtend for long-term memory.
- **Support for Multimodal Models**: Evaluate models integrating text, code, and other modalities.

## Contact

For inquiries or feedback, please [open an issue](https://github.com/zhimin-z/SE-Arena/issues/new) in this repository. We welcome your contributions and suggestions!
