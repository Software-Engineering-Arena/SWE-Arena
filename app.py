import dotenv
import evalica
import gitlab
import io
import json
import os
import random
import re
import threading

import gradio as gr
import pandas as pd

from datetime import datetime
from github import Github
from gradio_leaderboard import Leaderboard
from huggingface_hub import upload_file, hf_hub_download, HfFolder, HfApi
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI Client
api_key = os.getenv("API_KEY")
base_url = "https://api.pandalla.ai/v1"
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# Timeout in seconds for model responses
TIMEOUT = 90

# Hint string constant
SHOW_HINT_STRING = True  # Set to False to hide the hint string altogether
HINT_STRING = "Once signed in, your votes will be recorded securely."

# Load context length limits
with open("context_window.json", "r") as file:
    context_window = json.load(file)

# Get list of available models
available_models = list(context_window.keys())


def fetch_github_content(url):
    """Fetch detailed content from a GitHub URL using PyGithub."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set.")
        return None

    g = Github(token)

    try:
        match = re.match(
            r"https?://github\.com/([^/]+)/([^/]+)/(commit|pull|issues|discussions)/([a-z0-9]+)",
            url,
        )

        if not match:
            repo_part = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?", url)
            if repo_part:
                owner, repo = repo_part.groups()
                repo = g.get_repo(f"{owner}/{repo}")
                try:
                    readme = repo.get_readme()
                    return readme.decoded_content.decode()
                except:
                    return repo.description
            return None

        owner, repo, category, identifier = match.groups()
        repo = g.get_repo(f"{owner}/{repo}")

        if category == "commit":
            commit = repo.get_commit(identifier)
            return commit.__dict__

        elif category in ["pull", "issues"]:
            obj = (
                repo.get_pull(int(identifier))
                if category == "pull"
                else repo.get_issue(int(identifier))
            )
        return obj.__dict__

    except Exception as e:
        print(f"GitHub API error: {e}")
        return None


def fetch_gitlab_content(url):
    """Fetch content from GitLab URL using python-gitlab."""
    token = os.getenv("GITLAB_TOKEN")
    if not token:
        print("GITLAB_TOKEN not set.")
        return None
    gl = gitlab.Gitlab(private_token=token)

    try:
        match = re.match(
            r"https?://gitlab\.com/([^/]+)/([^/]+)/-/?(commit|merge_requests|issues)/([^/]+)",
            url,
        )
        if not match:
            repo_part = re.match(r"https?://gitlab\.com/([^/]+)/([^/]+)/?", url)
            if repo_part:
                owner, repo = repo_part.groups()
                project = gl.projects.get(f"{owner}/{repo}")
                try:
                    readme = project.files.get(file_path="README.md", ref="master")
                    return readme.decode()
                except gitlab.exceptions.GitlabGetError:
                    return project.description
            return None

        owner, repo, category, identifier = match.groups()
        project = gl.projects.get(f"{owner}/{repo}")

        if category == "commit":
            commit = project.commits.get(identifier)
            return commit.__dict__

        elif category == "merge_requests":
            merge_request = project.mergerequests.get(int(identifier))
            return merge_request.__dict__

        elif category == "issues":
            issue = project.issues.get(int(identifier))
            return issue.__dict__

    except Exception as e:
        print(f"GitLab API error: {e}")
        return None


def fetch_huggingface_content(url):
    """Fetch detailed content from a Hugging Face URL using huggingface_hub API."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set.")
        return None

    api = HfApi(token=token)

    try:
        if "/commit/" in url:
            commit_hash = url.split("/commit/")[-1]
            repo_id = url.split("/commit/")[0].split("huggingface.co/")[-1]
            commits = api.list_repo_commits(repo_id=repo_id, revision=commit_hash)
            if commits:
                commit = commits[0]
                return commit.__dict__
            return None

        elif "/discussions/" in url:
            discussion_num = int(url.split("/discussions/")[-1])
            repo_id = url.split("/discussions/")[0].split("/huggingface.co/")[-1]
            discussion = api.get_discussion_details(
                repo_id=repo_id, discussion_num=discussion_num
            )
            return discussion.__dict__

        else:
            repo_id = url.split("huggingface.co/")[-1]
            repo_info = api.repo_info(repo_id=repo_id)
            return repo_info.__dict__

    except Exception as e:
        print(f"Hugging Face API error: {e}")
    return None


def fetch_url_content(url):
    """Main URL content fetcher that routes to platform-specific handlers."""
    try:
        if "github.com" in url:
            return fetch_github_content(url)
        elif "gitlab.com" in url:
            return fetch_gitlab_content(url)
        elif "huggingface.co" in url:
            return fetch_huggingface_content(url)
    except Exception as e:
        print(f"Error fetching URL content: {e}")
    return ""


# Truncate prompt
def truncate_prompt(model_alias, models, conversation_state):
    """
    Truncate the conversation history and user input to fit within the model's context window.

    Args:
        model_alias (str): Alias for the model being used (i.e., "left", "right").
        models (dict): Dictionary mapping model aliases to their names.
        conversation_state (dict): State containing the conversation history for all models.

    Returns:
        str: Truncated conversation history and user input.
    """
    # Get the full conversation history for the model
    full_conversation = conversation_state[f"{model_alias}_chat"]

    # Get the context length for the model
    context_length = context_window[models[model_alias]]

    # Single loop to handle both FIFO removal and content truncation
    while len(json.dumps(full_conversation)) > context_length:
        # If we have more than one message, remove the oldest (FIFO)
        if len(full_conversation) > 1:
            full_conversation.pop(0)
        # If only one message remains, truncate its content
        else:
            current_length = len(json.dumps(full_conversation))
            # Calculate how many characters we need to remove
            excess = current_length - context_length
            # Add a buffer to ensure we remove enough (accounting for JSON encoding)
            truncation_size = min(excess + 10, len(full_conversation[0]["content"]))

            if truncation_size <= 0:
                break  # Can't truncate further

            # Truncate the content from the end to fit
            full_conversation[0]["content"] = full_conversation[0]["content"][
                :-truncation_size
            ]

    return full_conversation


def chat_with_models(model_alias, models, conversation_state, timeout=TIMEOUT):
    truncated_input = truncate_prompt(model_alias, models, conversation_state)
    response_event = threading.Event()  # Event to signal response completion
    model_response = {"content": None, "error": None}

    def request_model_response():
        try:
            request_params = {"model": models[model_alias], "messages": truncated_input}
            response = openai_client.chat.completions.create(**request_params)
            model_response["content"] = response.choices[0].message.content
        except Exception as e:
            model_response["error"] = (
                f"{models[model_alias]} model is not available. Error: {e}"
            )
        finally:
            response_event.set()  # Signal that the response is completed

    # Start the model request in a separate thread
    response_thread = threading.Thread(target=request_model_response)
    response_thread.start()

    # Wait for the specified timeout
    response_event_occurred = response_event.wait(timeout)

    if not response_event_occurred:
        raise TimeoutError(
            f"The {model_alias} model did not respond within {timeout} seconds."
        )
    elif model_response["error"]:
        raise Exception(model_response["error"])
    else:
        # Get the full conversation history for the model
        model_key = f"{model_alias}_chat"

        # Add the model's response to the conversation state
        conversation_state[model_key].append(
            {"role": "assistant", "content": model_response["content"]}
        )

        # Format the complete conversation history with different colors
        formatted_history = format_conversation_history(conversation_state[model_key][1:])

        return formatted_history


def format_conversation_history(conversation_history):
    """
    Format the conversation history with different colors for user and model messages.

    Args:
        conversation_history (list): List of conversation messages with role and content.

    Returns:
        str: Markdown formatted conversation history.
    """
    formatted_text = ""

    for message in conversation_history:
        if message["role"] == "user":
            # Format user messages with blue text
            formatted_text += f"<div style='color: #0066cc; background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>User:</strong> {message['content']}</div>\n\n"
        else:
            # Format assistant messages with dark green text
            formatted_text += f"<div style='color: #006633; background-color: #f0fff0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Model:</strong> {message['content']}</div>\n\n"

    return formatted_text


def save_content_to_hf(feedback_data, repo_name):
    """
    Save feedback content to Hugging Face repository organized by quarter.
    """
    # Serialize the content to JSON and encode it as bytes
    json_content = json.dumps(feedback_data, indent=4).encode("utf-8")

    # Create a binary file-like object
    file_like_object = io.BytesIO(json_content)

    # Get the current year and quarter
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    year_quarter = f"{now.year}_Q{quarter}"
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Define the path in the repository
    filename = f"{year_quarter}/{timestamp}.json"

    # Ensure the user is authenticated with HF
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Please log in to Hugging Face using `huggingface-cli login`.")

    # Upload to Hugging Face repository
    upload_file(
        path_or_fileobj=file_like_object,
        path_in_repo=filename,
        repo_id=repo_name,
        repo_type="dataset",
        use_auth_token=token,
    )


def load_content_from_hf(repo_name="SE-Arena/votes"):
    """
    Read feedback content from a Hugging Face repository based on the current quarter.

    Args:
        repo_name (str): Hugging Face repository name.

    Returns:
        list: Aggregated feedback data read from the repository.
    """
    feedback_data = []

    # Get the current year and quarter
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    year_quarter = f"{now.year}_Q{quarter}"

    try:
        api = HfApi()
        # List all files in the repository
        for file in api.list_repo_files(repo_id=repo_name, repo_type="dataset"):
            # Filter files by current year and month
            if year_quarter not in file:
                continue
            # Download and aggregate data
            local_path = hf_hub_download(
                repo_id=repo_name, filename=file, repo_type="dataset"
            )
            with open(local_path, "r") as f:
                data = json.load(f)
                # Add the timestamp to the data
                data["timestamp"] = file.split("/")[-1].split(".")[0]
                feedback_data.append(data)
        return feedback_data

    except:
        raise Exception("Error loading feedback data from Hugging Face repository.")


def get_leaderboard_data(feedback_entry=None):
    # Load feedback data from the Hugging Face repository
    feedback_data = load_content_from_hf()
    feedback_df = pd.DataFrame(feedback_data)

    # Load conversation data from the Hugging Face repository
    conversation_data = load_content_from_hf("SE-Arena/conversations")
    conversation_df = pd.DataFrame(conversation_data)

    # Concatenate the new feedback with the existing leaderboard data
    if feedback_entry is not None:
        feedback_df = pd.concat(
            [feedback_df, pd.DataFrame([feedback_entry])], ignore_index=True
        )

    if feedback_df.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Model",
                "Elo Score",
                "Consistency Score",
                "Average Win Rate",
                "Bradley-Terry Coefficient",
                "Eigenvector Centrality Value",
                "Newman Modularity Score",
                "PageRank Score",
            ]
        )

    # map vote to winner
    feedback_df["winner"] = feedback_df["winner"].map(
        {
            "left": evalica.Winner.X,
            "right": evalica.Winner.Y,
            "tie": evalica.Winner.Draw,
        }
    )

    # Calculate scores using various metrics
    avr_result = evalica.average_win_rate(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )
    bt_result = evalica.bradley_terry(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )
    newman_result = evalica.newman(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )
    eigen_result = evalica.eigen(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )
    elo_result = evalica.elo(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )
    pagerank_result = evalica.pagerank(
        feedback_df["left"], feedback_df["right"], feedback_df["winner"]
    )

    # Calculate consistency score as a pandas Series aligned with other metrics
    cs_result = pd.Series(
        "N/A", index=elo_result.scores.index
    )  # Initialize with zeros using same index

    # Loop through models and update values
    for model in cs_result.index:
        # Filter self-matches for this model
        self_matches = feedback_df[
            (feedback_df["left"] == model) & (feedback_df["right"] == model)
        ]
        totals = len(self_matches)

        if totals:
            # Count non-draw outcomes (wins or losses)
            cs_result[model] = round(
                self_matches[self_matches["winner"] == evalica.Winner.Draw].shape[0]
                / totals,
                2,
            )

    # Combine all results into a single DataFrame
    leaderboard_data = pd.DataFrame(
        {
            "Model": elo_result.scores.index,
            "Elo Score": elo_result.scores.values,
            "Consistency Score": cs_result.values,
            "Average Win Rate": avr_result.scores.values,
            "Bradley-Terry Coefficient": bt_result.scores.values,
            "Eigenvector Centrality Value": eigen_result.scores.values,
            "Newman Modularity Score": newman_result.scores.values,
            "PageRank Score": pagerank_result.scores.values,
        }
    )

    # Round all numeric columns to two decimal places
    leaderboard_data = leaderboard_data.round(
        {
            "Elo Score": 2,
            "Average Win Rate": 2,
            "Bradley-Terry Coefficient": 2,
            "Eigenvector Centrality Value": 2,
            "Newman Modularity Score": 2,
            "PageRank Score": 2,
        }
    )

    # Add a Rank column based on Elo scores
    leaderboard_data["Rank"] = (
        leaderboard_data["Elo Score"].rank(ascending=False).astype(int)
    )

    # Place rank in the first column
    leaderboard_data = leaderboard_data[
        ["Rank"] + [col for col in leaderboard_data.columns if col != "Rank"]
    ]
    return leaderboard_data


# Function to enable or disable submit buttons based on textbox content
def toggle_submit_button(text):
    if not text or text.strip() == "":
        return gr.update(interactive=False)  # Disable the button
    else:
        return gr.update(interactive=True)  # Enable the button


# Gradio Interface
with gr.Blocks() as app:
    user_authenticated = gr.State(False)
    models_state = gr.State({})
    conversation_state = gr.State({})

    with gr.Tab("🏆Leaderboard"):
        # Add title and description as a Markdown component
        leaderboard_intro = gr.Markdown(
            """
            # 🏆 FM4SE Leaderboard: Community-Driven Evaluation of Top Foundation Models (FMs) in Software Engineering (SE) Tasks
            The SE Arena is an open-source platform designed to evaluate foundation models through human preference, fostering transparency and collaboration. This platform aims to empower the SE community to assess and compare the performance of leading FMs in related tasks. For technical details, check out our [paper](https://arxiv.org/abs/2502.01860).
            """,
            elem_classes="leaderboard-intro",
        )
        # Initialize the leaderboard with the DataFrame containing the expected columns
        leaderboard_component = Leaderboard(
            value=get_leaderboard_data(),
            select_columns=[
                "Rank",
                "Model",
                "Elo Score",
                "Consistency Score",
            ],
            search_columns=["Model"],
            filter_columns=[
                "Elo Score",
                "Consistency Score",
                "Average Win Rate",
                "Bradley-Terry Coefficient",
                "Eigenvector Centrality Value",
                "Newman Modularity Score",
                "PageRank Score",
            ],
        )
        # Add a citation block in Markdown
        citation_component = gr.Markdown(
            """
            Made with ❤️ for SE Arena. If this work is useful to you, please consider citing:
            ```
            @inproceedings{zhao2025se,
            title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
            author={Zhao, Zhimin},
            booktitle={ACM international conference on AI Foundation Models and Software Engineering},
            year={2025}}
            ```
            """
        )
    with gr.Tab("⚔️Arena"):
        # Add title and description as a Markdown component
        arena_intro = gr.Markdown(
            f"""
            # ⚔️ SE Arena: Explore and Test Top FMs with SE Tasks by Community Voting

            ## 📜How It Works
            - **Blind Comparison**: Submit a SE-related query to two anonymous FMs randomly selected from up to {len(available_models)} top models from OpenAI, Gemini, Grok, Claude, Deepseek, Qwen, Llama, Mistral, and others.
            - **Interactive Voting**: Engage in multi-turn dialogues with both FMs and compare their responses. You can continue the conversation until you confidently choose the better model.
            - **Fair Play Rules**: Votes are counted only if FM identities remain anonymous. Revealing a FM's identity disqualifies the session.

            **Note:** Due to budget constraints, responses that take longer than {TIMEOUT} seconds to generate will be discarded.
            """,
            elem_classes="arena-intro",
        )
        # Add Hugging Face Sign In button and message
        with gr.Row():
            # Define the markdown text with or without the hint string
            markdown_text = "## Please sign in first to vote!"
            if SHOW_HINT_STRING:
                markdown_text += f"\n{HINT_STRING}"
            hint_markdown = gr.Markdown(markdown_text, elem_classes="markdown-text")
            login_button = gr.Button(
                "Sign in with Hugging Face", elem_id="oauth-button"
            )

        guardrail_message = gr.Markdown("", visible=False, elem_id="guardrail-message")

        # NEW: Add a textbox for the repository URL above the user prompt
        repo_url = gr.Textbox(
            show_label=False,
            placeholder="Optional: Enter the URL of a repository (GitHub, GitLab, Hugging Face), issue, commit, or pull request.",
            lines=1,
            interactive=False,
        )

        # Components with initial non-interactive state
        shared_input = gr.Textbox(
            show_label=False,
            placeholder="Enter your query for both models here.",
            lines=2,
            interactive=False,  # Initially non-interactive
        )
        send_first = gr.Button(
            "Submit", visible=True, interactive=False
        )  # Initially non-interactive

        # Add event listener to shared_input to toggle send_first button
        shared_input.change(
            fn=toggle_submit_button, inputs=shared_input, outputs=send_first
        )

        user_prompt_md = gr.Markdown(value="", visible=False)

        with gr.Column():
            shared_input
            user_prompt_md

        with gr.Row():
            response_a_title = gr.Markdown(value="", visible=False)
            response_b_title = gr.Markdown(value="", visible=False)

        with gr.Row():
            response_a = gr.Markdown(label="Response from Model A")
            response_b = gr.Markdown(label="Response from Model B")

        # Add a popup component for timeout notification
        with gr.Row(visible=False) as timeout_popup:
            timeout_message = gr.Markdown(
                "### Timeout\n\nOne of the models did not respond within 1 minute. Please try again."
            )
            close_popup_btn = gr.Button("Okay")

        def close_timeout_popup():
            # Re-enable or disable the submit buttons based on the current textbox content
            shared_input_state = gr.update(interactive=True)
            send_first_state = toggle_submit_button(shared_input.value)

            model_a_input_state = gr.update(interactive=True)
            model_a_send_state = toggle_submit_button(model_a_input.value)

            model_b_input_state = gr.update(interactive=True)
            model_b_send_state = toggle_submit_button(model_b_input.value)

            # Keep repo_url in sync with shared_input
            repo_url_state = gr.update(interactive=True)

            return (
                gr.update(visible=False),  # Hide the timeout popup
                shared_input_state,  # Update shared_input
                send_first_state,  # Update send_first button
                model_a_input_state,  # Update model_a_input
                model_a_send_state,  # Update model_a_send button
                model_b_input_state,  # Update model_b_input
                model_b_send_state,  # Update model_b_send button
                repo_url_state,  # Update repo_url button
            )

        # Multi-round inputs, initially hidden
        with gr.Row(visible=False) as multi_round_inputs:
            model_a_input = gr.Textbox(label="Model A Input", lines=1)
            model_a_send = gr.Button(
                "Send to Model A", interactive=False
            )  # Initially disabled

            model_b_input = gr.Textbox(label="Model B Input", lines=1)
            model_b_send = gr.Button(
                "Send to Model B", interactive=False
            )  # Initially disabled

        # Add event listeners to model_a_input and model_b_input to toggle their submit buttons
        model_a_input.change(
            fn=toggle_submit_button, inputs=model_a_input, outputs=model_a_send
        )

        model_b_input.change(
            fn=toggle_submit_button, inputs=model_b_input, outputs=model_b_send
        )

        close_popup_btn.click(
            close_timeout_popup,
            inputs=[],
            outputs=[
                timeout_popup,
                shared_input,
                send_first,
                model_a_input,
                model_a_send,
                model_b_input,
                model_b_send,
                repo_url,
            ],
        )

        def guardrail_check_se_relevance(user_input):
            """
            Use gpt-4o-mini to check if the user input is SE-related.
            Return True if it is SE-related, otherwise False.
            """
            # Example instructions for classification — adjust to your needs
            system_message = {
                "role": "system",
                "content": (
                    "You are a classifier that decides if a user's question is relevant to software engineering. "
                    "If the question is about software engineering concepts, tools, processes, or code, respond with 'Yes'. "
                    "Otherwise, respond with 'No'."
                ),
            }
            user_message = {"role": "user", "content": user_input}

            try:
                # Make the chat completion call
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=[system_message, user_message]
                )
                classification = response.choices[0].message.content.strip().lower()
                # Check if the LLM responded with 'Yes'
                return classification.lower().startswith("yes")
            except Exception as e:
                print(f"Guardrail check failed: {e}")
                # If there's an error, you might decide to fail open (allow) or fail closed (block).
                # Here we default to fail open, but you can change as needed.
                return True

        def disable_first_submit_ui():
            """First function to immediately disable UI elements"""
            return (
                # [0] guardrail_message: hide
                gr.update(visible=False),
                # [1] shared_input: disable but keep visible
                gr.update(interactive=False),
                # [2] repo_url: disable but keep visible
                gr.update(interactive=False),
                # [3] send_first: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        # Function to update model titles and responses
        def update_model_titles_and_responses(
            repo_url, user_input, models_state, conversation_state
        ):
            # Guardrail check first
            if not repo_url and not guardrail_check_se_relevance(user_input):
                # Return updates to show the guardrail message and re-enable UI
                return (
                    # [0] guardrail_message: Show guardrail message
                    gr.update(
                        value="### Oops! Try asking something about software engineering. Thanks!",
                        visible=True,
                    ),
                    # [1] shared_input: clear and re-enable
                    gr.update(value="", interactive=True, visible=True),
                    # [2] repo_url: clear and re-enable
                    gr.update(value="", interactive=True, visible=True),
                    # [3] user_prompt_md: clear and hide
                    gr.update(value="", visible=False),
                    # [4] response_a_title: clear and hide
                    gr.update(value="", visible=False),
                    # [5] response_b_title: clear and hide
                    gr.update(value="", visible=False),
                    # [6] response_a: clear response
                    gr.update(value=""),
                    # [7] response_b: clear response
                    gr.update(value=""),
                    # [8] multi_round_inputs: hide
                    gr.update(visible=False),
                    # [9] vote_panel: hide
                    gr.update(visible=False),
                    # [10] send_first: re-enable button with original text
                    gr.update(visible=True, interactive=True, value="Submit"),
                    # [11] feedback: enable the selection
                    gr.update(interactive=True),
                    # [12] models_state: pass state as-is
                    models_state,
                    # [13] conversation_state: pass state as-is
                    conversation_state,
                    # [14] timeout_popup: hide
                    gr.update(visible=False),
                    # [15] model_a_send: disable
                    gr.update(interactive=False),
                    # [16] model_b_send: disable
                    gr.update(interactive=False),
                    # [17] thanks_message: hide
                    gr.update(visible=False),
                )

            # Fetch repository info if a URL is provided
            repo_info = fetch_url_content(repo_url)
            combined_user_input = (
                f"Context: {repo_info}\n\nInquiry: {user_input}"
                if repo_info
                else user_input
            )

            # Randomly select two models for the comparison
            selected_models = [random.choice(available_models) for _ in range(2)]
            models = {"left": selected_models[0], "right": selected_models[1]}

            # Create a copy to avoid modifying the original
            conversations = models.copy()
            conversations.update({
                "url": repo_url,
                "left_chat": [{"role": "user", "content": combined_user_input}],
                "right_chat": [{"role": "user", "content": combined_user_input}]
            })

            # Clear previous states
            models_state.clear()
            conversation_state.clear()

            # Update the states
            models_state.update(models)
            conversation_state.update(conversations)

            try:
                response_a = chat_with_models("left", models_state, conversation_state)
                response_b = chat_with_models("right", models_state, conversation_state)
            except TimeoutError as e:
                # Handle timeout by resetting components and showing a popup.
                return (
                    # [0] guardrail_message: hide
                    gr.update(visible=False),
                    # [1] shared_input: re-enable and clear
                    gr.update(value="", interactive=True, visible=True),
                    # [2] repo_url: re-enable and clear
                    gr.update(value="", interactive=True, visible=True),
                    # [3] user_prompt_md: hide
                    gr.update(value="", visible=False),
                    # [4] response_a_title: hide
                    gr.update(value="", visible=False),
                    # [5] response_b_title: hide
                    gr.update(value="", visible=False),
                    # [6] response_a: clear
                    gr.update(value=""),
                    # [7] response_b: clear
                    gr.update(value=""),
                    # [8] multi_round_inputs: hide
                    gr.update(visible=False),
                    # [9] vote_panel: hide
                    gr.update(visible=False),
                    # [10] send_first: re-enable with original text
                    gr.update(visible=True, interactive=True, value="Submit"),
                    # [11] feedback: disable
                    gr.update(interactive=False),
                    # [12] models_state: pass state as-is
                    models_state,
                    # [13] conversation_state: pass state as-is
                    conversation_state,
                    # [14] timeout_popup: show popup
                    gr.update(visible=True),
                    # [15] model_a_send: disable
                    gr.update(interactive=False),
                    # [16] model_b_send: disable
                    gr.update(interactive=False),
                    # [17] thanks_message: hide
                    gr.update(visible=False),
                )
            except Exception as e:
                raise gr.Error(str(e))

            # Determine the initial state of the multi-round send buttons
            model_a_send_state = toggle_submit_button("")
            model_b_send_state = toggle_submit_button("")
            display_content = f"### Your Query:\n\n{user_input}"
            if repo_info:
                display_content += f"\n\n### Repo-related URL:\n\n{repo_url}"

            # Return the updates for all 18 outputs.
            return (
                # [0] guardrail_message: hide (since no guardrail issue)
                gr.update(visible=False),
                # [1] shared_input: re-enable but hide
                gr.update(interactive=True, visible=False),
                # [2] repo_url: re-enable but hide
                gr.update(interactive=True, visible=False),
                # [3] user_prompt_md: display the user's query
                gr.update(value=display_content, visible=True),
                # [4] response_a_title: show title for Model A
                gr.update(value="### Model A:", visible=True),
                # [5] response_b_title: show title for Model B
                gr.update(value="### Model B:", visible=True),
                # [6] response_a: display Model A response
                gr.update(value=response_a),
                # [7] response_b: display Model B response
                gr.update(value=response_b),
                # [8] multi_round_inputs: show the input section for multi-round dialogues
                gr.update(visible=True),
                # [9] vote_panel: show vote panel
                gr.update(visible=True),
                # [10] send_first: hide the submit button but restore label
                gr.update(visible=False, value="Submit"),
                # [11] feedback: enable the feedback selection
                gr.update(interactive=True),
                # [12] models_state: pass updated models_state
                models_state,
                # [13] conversation_state: pass updated conversation_state
                conversation_state,
                # [14] timeout_popup: hide any timeout popup if visible
                gr.update(visible=False),
                # [15] model_a_send: set state of the model A send button
                model_a_send_state,
                # [16] model_b_send: set state of the model B send button
                model_b_send_state,
                # [17] thanks_message: hide the thank-you message
                gr.update(visible=False),
            )

        # Feedback panel, initially hidden
        with gr.Row(visible=False) as vote_panel:
            feedback = gr.Radio(
                choices=["Model A", "Model B", "Can't Decide"],
                label="Which model do you prefer?",
                value="Can't Decide",
                interactive=False,  # Initially not interactive
            )
            submit_feedback_btn = gr.Button("Submit Feedback", interactive=False)

        thanks_message = gr.Markdown(
            value="## Thanks for your vote!", visible=False
        )  # Add thank you message

        def hide_thanks_message():
            return gr.update(visible=False)

        # Function to handle login
        def handle_login():
            """
            Handle user login using Hugging Face OAuth with automatic redirection.
            """
            try:
                # Use Hugging Face OAuth to initiate login
                HfApi()
                token = HfFolder.get_token()
                if not token:
                    raise Exception("Authentication token not found.")

                # If token is successfully retrieved, update the interface state
                return (
                    gr.update(visible=False),  # Hide the login button
                    gr.update(interactive=True),  # repo_url -> Enable in sync
                    gr.update(interactive=True),  # Enable shared_input
                    gr.update(
                        interactive=False
                    ),  # Keep send_first button disabled initially
                    gr.update(interactive=True),  # Enable feedback radio buttons
                    gr.update(interactive=True),  # Enable submit_feedback_btn
                    gr.update(visible=False),  # Hide the hint string
                )
            except Exception as e:
                # Handle login failure
                print(f"Login failed: {e}")
                return (
                    gr.update(visible=True),  # Keep the login button visible
                    gr.update(interactive=False),  # repo_url -> disable if login failed
                    gr.update(interactive=False),  # Keep shared_input disabled
                    gr.update(interactive=False),  # Keep send_first disabled
                    gr.update(
                        interactive=False
                    ),  # Keep feedback radio buttons disabled
                    gr.update(interactive=False),  # Keep submit_feedback_btn disabled
                    gr.update(visible=True),  # Show the hint string
                )

        # Handle the login button click
        login_button.click(
            handle_login,
            inputs=[],
            outputs=[
                login_button,  # Hide the login button after successful login
                repo_url,  # Keep this in sync with shared_input
                shared_input,  # Enable shared_input
                send_first,  # Enable send_first button
                feedback,  # Enable feedback radio buttons
                submit_feedback_btn,  # Enable submit_feedback_btn
                hint_markdown,  # Hide the hint string
            ],
        )

        # First round handling
        send_first.click(
            fn=hide_thanks_message, inputs=[], outputs=[thanks_message]
        ).then(
            fn=disable_first_submit_ui,  # First disable UI
            inputs=[],
            outputs=[
                guardrail_message,
                shared_input,
                repo_url,
                send_first,  # Just the essential UI elements to update immediately
            ],
        ).then(
            fn=update_model_titles_and_responses,  # Then do the actual processing
            inputs=[repo_url, shared_input, models_state, conversation_state],
            outputs=[
                guardrail_message,
                shared_input,
                repo_url,
                user_prompt_md,
                response_a_title,
                response_b_title,
                response_a,
                response_b,
                multi_round_inputs,
                vote_panel,
                send_first,
                feedback,
                models_state,
                conversation_state,
                timeout_popup,
                model_a_send,
                model_b_send,
                thanks_message,
            ],
        )

        def disable_model_a_ui():
            """First function to immediately disable model A UI elements"""
            return (
                # [0] model_a_input: disable
                gr.update(interactive=False),
                # [1] model_a_send: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        # Handle subsequent rounds
        def handle_model_a_send(user_input, models_state, conversation_state):
            try:
                conversation_state["left_chat"].append({"role": "user", "content": user_input})
                response = chat_with_models("left", models_state, conversation_state)
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_a_input
                    gr.update(
                        interactive=False, value="Send to Model A"
                    ),  # Reset button text
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=True),  # Re-enable model_a_input
                    gr.update(
                        interactive=True, value="Send to Model A"
                    ),  # Re-enable model_a_send button
                )
            except Exception as e:
                raise gr.Error(str(e))

        def disable_model_b_ui():
            """First function to immediately disable model B UI elements"""
            return (
                # [0] model_b_input: disable
                gr.update(interactive=False),
                # [1] model_b_send: disable and show loading state
                gr.update(interactive=False, value="Processing..."),
            )

        def handle_model_b_send(user_input, models_state, conversation_state):
            try:
                conversation_state["right_chat"].append({"role": "user", "content": user_input})
                response = chat_with_models("right", models_state, conversation_state)
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_b_input
                    gr.update(
                        interactive=False, value="Send to Model B"
                    ),  # Reset button text
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=True),  # Re-enable model_b_input
                    gr.update(
                        interactive=True, value="Send to Model B"
                    ),  # Re-enable model_b_send button
                )
            except Exception as e:
                raise gr.Error(str(e))

        model_a_send.click(
            fn=disable_model_a_ui,  # First disable UI
            inputs=[],
            outputs=[model_a_input, model_a_send],
        ).then(
            fn=handle_model_a_send,  # Then do the actual processing
            inputs=[model_a_input, models_state, conversation_state],
            outputs=[
                response_a,
                conversation_state,
                timeout_popup,
                model_a_input,
                model_a_send,
            ],
        )
        model_b_send.click(
            fn=disable_model_b_ui,  # First disable UI
            inputs=[],
            outputs=[model_b_input, model_b_send],
        ).then(
            fn=handle_model_b_send,  # Then do the actual processing
            inputs=[model_b_input, models_state, conversation_state],
            outputs=[
                response_b,
                conversation_state,
                timeout_popup,
                model_b_input,
                model_b_send,
            ],
        )

        def submit_feedback(vote, models_state, conversation_state):
            # Map vote to actual model names
            match vote:
                case "Model A":
                    winner_model = "left"
                case "Model B":
                    winner_model = "right"
                case "Can't Decide":
                    winner_model = "tie"

            # Create feedback entry
            feedback_entry = {
                "left": models_state["left"],
                "right": models_state["right"],
                "winner": winner_model,
            }

            # Save feedback back to the Hugging Face dataset
            save_content_to_hf(feedback_entry, "SE-Arena/votes")

            conversation_state["right_chat"][0]["content"] = conversation_state[
                "right_chat"
            ][0]["content"].split("\n\nInquiry: ")[-1]
            conversation_state["left_chat"][0]["content"] = conversation_state[
                "left_chat"
            ][0]["content"].split("\n\nInquiry: ")[-1]

            # Save conversations back to the Hugging Face dataset
            save_content_to_hf(conversation_state, "SE-Arena/conversations")

            # Clear state
            models_state.clear()
            conversation_state.clear()

            # Adjust output count to match the interface definition
            return (
                gr.update(
                    value="", interactive=True, visible=True
                ),  # [0] Clear shared_input textbox
                gr.update(
                    value="", interactive=True, visible=True
                ),  # [1] Clear repo_url textbox
                gr.update(
                    value="", visible=False
                ),  # [2] Hide user_prompt_md markdown component
                gr.update(
                    value="", visible=False
                ),  # [3] Hide response_a_title markdown component
                gr.update(
                    value="", visible=False
                ),  # [4] Hide response_b_title markdown component
                gr.update(value=""),  # [5] Clear Model A response markdown component
                gr.update(value=""),  # [6] Clear Model B response markdown component
                gr.update(visible=False),  # [7] Hide multi_round_inputs row
                gr.update(visible=False),  # [8] Hide vote_panel row
                gr.update(
                    value="Submit", interactive=True, visible=True
                ),  # [9] Reset send_first button
                gr.update(
                    value="Can't Decide", interactive=True
                ),  # [10] Reset feedback radio selection
                get_leaderboard_data(feedback_entry),  # [11] Updated leaderboard data
                gr.update(
                    visible=True
                ),  # [12] Show the thanks_message markdown component
            )

        # Update the click event for the submit feedback button
        submit_feedback_btn.click(
            submit_feedback,
            inputs=[feedback, models_state, conversation_state],
            outputs=[
                shared_input,  # Reset shared_input
                repo_url,  # Show the repo-related URL message
                user_prompt_md,  # Hide user_prompt_md
                response_a_title,  # Hide Model A title
                response_b_title,  # Hide Model B title
                response_a,  # Clear Model A response
                response_b,  # Clear Model B response
                multi_round_inputs,  # Hide multi-round input section
                vote_panel,  # Hide vote panel
                send_first,  # Reset and update send_first button
                feedback,  # Reset feedback selection
                leaderboard_component,  # Update leaderboard data dynamically
                thanks_message,  # Show the "Thanks for your vote!" message
            ],
        )

        # Add Terms of Service at the bottom
        terms_of_service = gr.Markdown(
            """
            ## Terms of Service

            Users are required to agree to the following terms before using the service:
            
            - The service is a **research preview**. It only provides limited safety measures and may generate offensive content.
            - It must not be used for any **illegal, harmful, violent, racist, or sexual** purposes.
            - Please do not upload any **private** information.
            - The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or a similar license.
            """
        )

    app.launch()
