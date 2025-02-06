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
TIMEOUT = 60

# Hint string constant
SHOW_HINT_STRING = True  # Set to False to hide the hint string altogether
HINT_STRING = "Once signed in, your votes will be recorded securely."

# Load context length limits
with open("context_window.json", "r") as file:
    context_window = json.load(file)

# Get list of available models
available_models = list(context_window.keys())
if len(available_models) < 2:
    raise ValueError(
        "Insufficient models in context_window.json. At least two are required."
    )

# Initialize global variables
models_state = {}
conversation_state = {}


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
    return None


# Truncate prompt
def truncate_prompt(user_input, model_alias, models, conversation_state):
    """
    Truncate the conversation history and user input to fit within the model's context window.

    Args:
        user_input (str): The latest input from the user.
        model_alias (str): Alias for the model being used (e.g., "Model A", "Model B").
        models (dict): Dictionary mapping model aliases to their names.
        conversation_state (dict): State containing the conversation history for all models.

    Returns:
        str: Truncated conversation history and user input.
    """
    model_name = models[model_alias]
    context_length = context_window.get(model_name, 4096)

    # Get the full conversation history for the model
    history = conversation_state.get(model_name, [])
    full_conversation = [
        {"role": msg["role"], "content": msg["content"]} for msg in history
    ]
    full_conversation.append({"role": "user", "content": user_input})

    # Convert to JSON string for accurate length measurement
    json_conversation = json.dumps(full_conversation)

    if len(json_conversation) <= context_length:
        # If the full conversation fits, return it as-is
        return full_conversation

    # Truncate based on the current round
    if not history:  # First round, truncate FILO
        while len(json.dumps(full_conversation)) > context_length:
            full_conversation.pop(0)  # Remove from the start
    else:  # Subsequent rounds, truncate FIFO
        while len(json.dumps(full_conversation)) > context_length:
            full_conversation.pop(-1)  # Remove from the end

    return full_conversation


def chat_with_models(
    user_input, model_alias, models, conversation_state, timeout=TIMEOUT
):
    model_name = models[model_alias]
    truncated_input = truncate_prompt(
        user_input, model_alias, models, conversation_state
    )
    conversation_state.setdefault(model_name, []).append(
        {"role": "user", "content": user_input}
    )

    response_event = threading.Event()  # Event to signal response completion
    model_response = {"content": None, "error": None}

    def request_model_response():
        try:
            request_params = {
                "model": model_name,
                "messages": truncated_input,
                "temperature": 0,
            }
            response = openai_client.chat.completions.create(**request_params)
            model_response["content"] = response.choices[0].message.content
        except Exception as e:
            model_response["error"] = f"{model_name} model is not available. Error: {e}"
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
        formatted_response = f"```\n{model_response['content']}\n```"
        conversation_state[model_name].append(
            {"role": "assistant", "content": model_response["content"]}
        )
        return formatted_response


def save_content_to_hf(content, repo_name):
    """
    Save feedback content to Hugging Face repository organized by month and year.

    Args:
        content (dict): Feedback data to be saved.
        month_year (str): Year and month string in the format "YYYY_MM".
        repo_name (str): Hugging Face repository name.
    """
    # Ensure the user is authenticated with HF
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Please log in to Hugging Face using `huggingface-cli login`.")

    # Serialize the content to JSON and encode it as bytes
    json_content = json.dumps(content, indent=4).encode("utf-8")

    # Create a binary file-like object
    file_like_object = io.BytesIO(json_content)

    # Get the current year and month
    month_year = datetime.now().strftime("%Y_%m")
    day_hour_minute_second = datetime.now().strftime("%d_%H%M%S")

    # Define the path in the repository
    filename = f"{month_year}/{day_hour_minute_second}.json"

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
    Read feedback content from a Hugging Face repository based on the current month and year.

    Args:
        repo_name (str): Hugging Face repository name.

    Returns:
        list: Aggregated feedback data read from the repository.
    """

    # Get the current year and month
    year_month = datetime.now().strftime("%Y_%m")
    feedback_data = []

    try:
        api = HfApi()
        # List all files in the repository
        repo_files = api.list_repo_files(repo_id=repo_name, repo_type="dataset")

        # Filter files by current year and month
        feedback_files = [file for file in repo_files if year_month in file]

        if not feedback_files:
            raise FileNotFoundError(
                f"No feedback files found for {year_month} in {repo_name}."
            )

        # Download and aggregate data
        for file in feedback_files:
            local_path = hf_hub_download(
                repo_id=repo_name, filename=file, repo_type="dataset"
            )
            with open(local_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    feedback_data.extend(data)
                elif isinstance(data, dict):
                    feedback_data.append(data)

        return feedback_data

    except:
        raise Exception("Error loading feedback data from Hugging Face repository.")


def get_leaderboard_data():
    # Load feedback data from the Hugging Face repository
    try:
        feedback_data = load_content_from_hf()
        feedback_df = pd.DataFrame(feedback_data)
    except:
        # If no feedback exists, return an empty DataFrame
        return pd.DataFrame(
            columns=[
                "Rank",
                "Model",
                "Elo Score",
                "Average Win Rate",
                "Bradley-Terry Coefficient",
                "Eigenvector Centrality Value",
                "Newman Modularity Score",
                "PageRank Score",
            ]
        )

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

    # Combine all results into a single DataFrame
    ranking_df = pd.DataFrame(
        {
            "Model": elo_result.scores.index,
            "Elo Score": elo_result.scores.values,
            "Average Win Rate": avr_result.scores.values * 100,
            "Bradley-Terry Coefficient": bt_result.scores.values,
            "Eigenvector Centrality Value": eigen_result.scores.values,
            "PageRank Score": pagerank_result.scores.values,
            "Newman Modularity Score": newman_result.scores.values,
        }
    )

    # Add a Rank column based on Elo scores
    ranking_df["Rank"] = (
        ranking_df["Elo Score"].rank(ascending=False, method="min").astype(int)
    )

    # Round all numeric columns to two decimal places
    ranking_df = ranking_df.round(
        {
            "Elo Score": 2,
            "Average Win Rate": 2,
            "Bradley-Terry Coefficient": 2,
            "Eigenvector Centrality Value": 2,
            "PageRank Score": 2,
            "Newman Modularity Score": 2,
        }
    )

    # Reorder columns to make 'Rank' the first column
    ranking_df = ranking_df.sort_values(by="Rank").reset_index(drop=True)

    ranking_df = ranking_df[
        [
            "Rank",
            "Model",
            "Elo Score",
            "Average Win Rate",
            "Bradley-Terry Coefficient",
            "Eigenvector Centrality Value",
            "Newman Modularity Score",
            "PageRank Score",
        ]
    ]

    return ranking_df


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
            # 🏆 Software Engineering (SE) Chatbot Leaderboard: Community-Driven Evaluation of Top SE Chatbots

            The SE Arena is an open-source platform designed to evaluate language models through human preference, fostering transparency and collaboration. Developed by researchers at [Software Analysis and Intelligence Lab (SAIL)](https://sail.cs.queensu.ca), the platform empowers the community to assess and compare the performance of leading foundation models in SE tasks. For technical details, check out our [paper](https://arxiv.org/abs/2502.01860).
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
                "Average Win Rate",
            ],
            search_columns=["Model"],
            filter_columns=[
                "Elo Score",
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
            [TODO]
            ```
            """
        )
    with gr.Tab("⚔️Arena"):
        # Add title and description as a Markdown component
        arena_intro = gr.Markdown(
            f"""
            # ⚔️ Software Engineering (SE) Arena: Explore and Test the Best SE Chatbots with Long-Context Interactions

            ## 📜How It Works
            - **Blind Comparison**: Submit a SE-related query to two anonymous chatbots randomly selected from up to {len(available_models)} top models, including ChatGPT, Gemini, Claude, Deepseek, Llama, and others.
            - **Interactive Voting**: Engage in multi-turn dialogues with both chatbots and compare their responses. You can continue the conversation until you confidently choose the better model.
            - **Fair Play Rules**: Votes are counted only if chatbot identities remain anonymous. Revealing a chatbot's identity disqualifies the session.

            **Note:** Due to budget constraints, responses that take longer than one minute to generate will be discarded.
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
        
        # NEW: Add a textbox for the repository URL above the user prompt
        repo_url = gr.Textbox(
            show_label=False,
            placeholder="Enter the repo-related URL here (optional)",
            lines=1,
            interactive=False,
        )
        
        # Components with initial non-interactive state
        shared_input = gr.Textbox(
            show_label=False,
            placeholder="Enter your query for both models here",
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

        # Function to update model titles and responses
        def update_model_titles_and_responses(
            repo_info, user_input, models_state, conversation_state
        ):
            # Combine repo-related information (if any) and user query into one prompt.
            combined_user_input = f"Repo-related Information: {fetch_url_content(repo_info)}\n\n{user_input}" if repo_info else user_input

            # Dynamically select two random models
            if len(available_models) < 2:
                raise ValueError(
                    "Insufficient models in context_window.json. At least two are required."
                )
            selected_models = random.sample(available_models, 2)
            models = {"Model A": selected_models[0], "Model B": selected_models[1]}

            # Update the states
            models_state.clear()
            models_state.update(models)
            conversation_state.clear()
            conversation_state.update({name: [] for name in models.values()})

            try:
                response_a = chat_with_models(
                    combined_user_input, "Model A", models_state, conversation_state
                )
                response_b = chat_with_models(
                    combined_user_input, "Model B", models_state, conversation_state
                )
            except TimeoutError as e:
                # Handle the timeout by resetting components, showing a popup, and disabling inputs
                return (
                    gr.update(
                        value="", interactive=False, visible=True
                    ),  # Disable shared_input
                    gr.update(
                        value="", interactive=False, visible=True
                    ),  # Disable repo_url
                    gr.update(value="", visible=False),  # Hide user_prompt_md
                    gr.update(value="", visible=False),  # Hide Model A title
                    gr.update(value="", visible=False),  # Hide Model B title
                    gr.update(value=""),  # Clear response from Model A
                    gr.update(value=""),  # Clear response from Model B
                    gr.update(visible=False),  # Hide multi-round inputs
                    gr.update(visible=False),  # Hide vote panel
                    gr.update(visible=True, interactive=False),  # Disable submit button
                    gr.update(interactive=False),  # Disable feedback selection
                    models_state,
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                )
            except Exception as e:
                raise gr.Error(str(e))

            # Determine the initial state of the multi-round send buttons
            model_a_send_state = toggle_submit_button("")
            model_b_send_state = toggle_submit_button("")

            return (
                gr.update(visible=False),  # Hide shared_input
                gr.update(visible=False),  # Hide repo_url the same way
                gr.update(
                    value=f"**Your Query:**\n\n{user_input}", visible=True
                ),  # Show user_prompt_md
                gr.update(value=f"### Model A:", visible=True),
                gr.update(value=f"### Model B:", visible=True),
                gr.update(value=response_a),  # Show Model A response
                gr.update(value=response_b),  # Show Model B response
                gr.update(visible=True),  # Show multi-round inputs
                gr.update(visible=True),  # Show vote panel
                gr.update(visible=False),  # Hide submit button
                gr.update(interactive=True),  # Enable feedback selection
                models_state,
                conversation_state,
                gr.update(visible=False),  # Hide the timeout popup if it was visible
                model_a_send_state,  # Set model_a_send button state
                model_b_send_state,  # Set model_b_send button state
                gr.update(
                    visible=False
                ),  # thanks_message - Make sure to return it as invisible here as well
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

                # Wait for user authentication and get the token
                print(
                    "Redirected to Hugging Face for authentication. Please complete the login."
                )
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
                    gr.update(interactive=False), # repo_url -> disable if login failed
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
                repo_url,      # Keep this in sync with shared_input
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
            fn=update_model_titles_and_responses,
            inputs=[repo_url, shared_input, models_state, conversation_state],
            outputs=[
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

        # Handle subsequent rounds
        def handle_model_a_send(user_input, models_state, conversation_state):
            try:
                response = chat_with_models(
                    user_input, "Model A", models_state, conversation_state
                )
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_a_input
                    gr.update(interactive=False),  # Disable model_a_send button
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=False),  # Disable model_a_input
                    gr.update(interactive=False),  # Disable model_a_send
                )
            except Exception as e:
                raise gr.Error(str(e))

        def handle_model_b_send(user_input, models_state, conversation_state):
            try:
                response = chat_with_models(
                    user_input, "Model B", models_state, conversation_state
                )
                # Clear the input box and disable the send button
                return (
                    response,
                    conversation_state,
                    gr.update(visible=False),
                    gr.update(
                        value="", interactive=True
                    ),  # Clear and enable model_b_input
                    gr.update(interactive=False),  # Disable model_b_send button
                )
            except TimeoutError as e:
                # Disable inputs when timeout occurs
                return (
                    gr.update(value=""),  # Clear response
                    conversation_state,
                    gr.update(visible=True),  # Show the timeout popup
                    gr.update(interactive=False),  # Disable model_b_input
                    gr.update(interactive=False),  # Disable model_b_send
                )
            except Exception as e:
                raise gr.Error(str(e))

        model_a_send.click(
            handle_model_a_send,
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
            handle_model_b_send,
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
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                "left": models_state["Model A"],
                "right": models_state["Model B"],
                "winner": winner_model,
                "timestamp": timestamp,
            }

            # Save feedback back to the Hugging Face dataset
            save_content_to_hf(feedback_entry, "SE-Arena/votes")

            # Save conversations back to the Hugging Face dataset
            save_content_to_hf(conversation_state, "SE-Arena/conversations")

            # Clear state
            models_state.clear()
            conversation_state.clear()

            # Recalculate leaderboard
            leaderboard_data = get_leaderboard_data()

            # Adjust output count to match the interface definition
            return (
                gr.update(
                    value="", interactive=True, visible=True
                ),  # Clear shared_input
                gr.update(
                    value="", interactive=True, visible=True
                ),  # Clear repo_url
                gr.update(value="", visible=False),  # Hide user_prompt_md
                gr.update(value="", visible=False),  # Hide response_a_title
                gr.update(value="", visible=False),  # Hide response_b_title
                gr.update(value=""),  # Clear Model A response
                gr.update(value=""),  # Clear Model B response
                gr.update(visible=False),  # Hide multi-round inputs
                gr.update(visible=False),  # Hide vote panel
                gr.update(
                    value="Submit", interactive=True, visible=True
                ),  # Update send_first button
                gr.update(
                    value="Can't Decide", interactive=True
                ),  # Reset feedback selection
                leaderboard_data,  # Updated leaderboard data
                gr.update(visible=True),  # Show the thanks message
                gr.update(value="", interactive=True, visible=True),  # Show the repo-related url message
            )

        # Update the click event for the submit feedback button
        submit_feedback_btn.click(
            submit_feedback,
            inputs=[feedback, models_state, conversation_state],
            outputs=[
                shared_input,  # Reset shared_input
                repo_url,   # Show the repo-related URL message
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
            - Please **do not upload any private information**.
            - The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or a similar license.
            """
        )

    app.launch()
