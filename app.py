import dotenv
import evalica
import io
import json
import os
import random
import threading

import aisuite as ai
import gradio as gr
import pandas as pd

from huggingface_hub import upload_file, hf_hub_download, HfFolder, HfApi
from datetime import datetime
from gradio_leaderboard import Leaderboard

# Load environment variables
dotenv.load_dotenv()

# Retrieve the secret from the environment
gcp_credentials = os.environ.get("GCP_CREDENTIALS")

# Write it to a file
credentials_path = (
    "/tmp/gcp_credentials.json"  # Ensure this path is secure and temporary
)
with open(credentials_path, "w") as f:
    f.write(gcp_credentials)

# Set the environment variable for GCP SDKs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Timeout in seconds for model response
TIMEOUT = 60  

# Initialize AISuite Client
client = ai.Client()

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

# Define functions here


# Truncate prompt
def truncate_prompt(prompt, model_alias, models):
    model_name = models[model_alias]
    context_length = context_window.get(model_name, 4096)
    while len(json.dumps({"role": "user", "content": prompt})) > context_length:
        prompt = prompt[:-10] if len(prompt) > 10 else prompt[:1]
    return prompt


def chat_with_models(user_input, model_alias, models, conversation_state, timeout=TIMEOUT):
    model_name = models[model_alias]
    truncated_input = truncate_prompt(user_input, model_alias, models)
    conversation_state.setdefault(model_name, []).append(
        {"role": "user", "content": user_input}
    )

    response_event = threading.Event()  # Event to signal response completion
    model_response = {"content": None, "error": None}

    def request_model_response():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": truncated_input}],
            )
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
        # Timeout occurred, raise a TimeoutError to be handled in the Gradio interface
        raise TimeoutError(
            f"The {model_alias} model did not respond within {timeout} seconds."
        )
    elif model_response["error"]:
        # An error occurred during model response
        raise Exception(model_response["error"])
    else:
        # Successful response
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
        print('here')
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
                "PageRank Score",
                "Newman Modularity Score",
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
            "PageRank Score",
            "Newman Modularity Score",
        ]
    ]

    return ranking_df


# Function to enable or disable submit buttons based on textbox content
def toggle_submit_button(text):
    if not text or text.strip() == "":
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


# Gradio Interface
with gr.Blocks() as app:
    user_authenticated = gr.State(False)
    models_state = gr.State({})
    conversation_state = gr.State({})

    with gr.Tab("🏆Leaderboard"):
        # Add title and description as a Markdown component
        leaderboard_intro = gr.Markdown(
            """
            # 🏆 Software Engineering Arena Leaderboard: Community-Driven Evaluation of Top SE Chatbots

            The Software Engineering (SE) Arena is an open-source platform designed to evaluate language models through human preference, fostering transparency and collaboration. Developed by researchers at [Software Analysis and Intelligence Lab (SAIL)](https://sail.cs.queensu.ca), the platform empowers the community to assess and compare the performance of leading foundation models in SE tasks. For technical details, check out our [paper](https://arxiv.org/abs/your-paper-link).
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
                "PageRank Score",
                "Newman Modularity Score",
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
            """
            # ⚔️ Software Engineering (SE) Arena: Explore and Test the Best SE Chatbots with Long-Context Interactions

            ## 📜How It Works
            - **Blind Comparison**: Submit any software engineering-related query to two anonymous chatbots, including top models like ChatGPT, Gemini, Claude, Llama, and others.
            - **Interactive Voting**: Engage in multi-turn dialogues and compare responses. Continue the conversation until you're confident in choosing the better model.
            - **Fair Play Rules**: Votes are valid only when chatbot identities remain anonymous—revealed identities disqualify the session.

            **Note:** Due to budget constraints, responses that take longer than one minute to generate will be discarded.
            """,
            elem_classes="arena-intro",
        )
        # Add Hugging Face Sign In button and message
        with gr.Row():
            # Define the markdown text with or without the hint string
            markdown_text = "## Please sign in using the button on the right to vote!"
            if SHOW_HINT_STRING:
                markdown_text += f"\n{HINT_STRING}"
            hint_markdown = gr.Markdown(markdown_text, elem_classes="markdown-text")
            login_button = gr.Button(
                "Sign in with Hugging Face", elem_id="oauth-button"
            )

        # Components with initial non-interactive state
        shared_input = gr.Textbox(
            label="Enter your prompt for both models",
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

            return (
                gr.update(visible=False),  # Hide the timeout popup
                shared_input_state,  # Update shared_input
                send_first_state,  # Update send_first button
                model_a_input_state,  # Update model_a_input
                model_a_send_state,  # Update model_a_send button
                model_b_input_state,  # Update model_b_input
                model_b_send_state,  # Update model_b_send button
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
            ],
        )

        # Function to update model titles and responses
        def update_model_titles_and_responses(
            user_input, models_state, conversation_state
        ):
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
                    user_input, "Model A", models_state, conversation_state
                )
                response_b = chat_with_models(
                    user_input, "Model B", models_state, conversation_state
                )
            except TimeoutError as e:
                # Handle the timeout by resetting components, showing a popup, and disabling inputs
                return (
                    gr.update(
                        value="", interactive=False, visible=True
                    ),  # Disable shared_input
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
                gr.update(
                    value=f"**Your Prompt:**\n\n{user_input}", visible=True
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
                    gr.update(interactive=True),  # Enable shared_input
                    gr.update(interactive=True),  # Enable send_first button
                    gr.update(interactive=True),  # Enable feedback radio buttons
                    gr.update(interactive=True),  # Enable submit_feedback_btn
                    gr.update(visible=False),  # Hide the hint string
                )
            except Exception as e:
                # Handle login failure
                print(f"Login failed: {e}")
                return (
                    gr.update(visible=True),  # Keep the login button visible
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
                shared_input,  # Enable shared_input
                send_first,  # Enable send_first button
                feedback,  # Enable feedback radio buttons
                submit_feedback_btn,  # Enable submit_feedback_btn
                hint_markdown,  # Hide the hint string
            ],
        )

        # First round handling
        send_first.click(
            update_model_titles_and_responses,
            inputs=[shared_input, models_state, conversation_state],
            outputs=[
                shared_input,  # shared_input
                user_prompt_md,  # user_prompt_md
                response_a_title,  # response_a_title
                response_b_title,  # response_b_title
                response_a,  # response_a
                response_b,  # response_b
                multi_round_inputs,  # multi_round_inputs
                vote_panel,  # vote_panel
                send_first,  # send_first
                feedback,  # feedback
                models_state,  # models_state
                conversation_state,  # conversation_state
                timeout_popup,  # timeout_popup
                model_a_send,  # model_a_send state
                model_b_send,  # model_b_send state
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
                ),  # Clear and show shared_input
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
            )

        # Update the click event for the submit feedback button
        submit_feedback_btn.click(
            submit_feedback,
            inputs=[feedback, models_state, conversation_state],
            outputs=[
                shared_input,  # Reset shared_input
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
            ],
        )

        # Add Terms of Service at the bottom
        terms_of_service = gr.Markdown(
            """
            ## Terms of Service

            Users are required to agree to the following terms before using the service:
            
            - The service is a **research preview**. It only provides limited safety measures and may generate offensive content.
            - It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
            - Please **do not upload any private information**.
            - The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or a similar license.
            """
        )

    app.launch()
