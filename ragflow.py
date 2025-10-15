import pickle
import random
import numpy as np
import json
import requests
import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from keras.models import load_model
from keras.utils import pad_sequences

# Set page configuration
st.set_page_config(
    page_title="Reflection Chatbot",
    page_icon="🤔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.user_input = ""
    st.session_state.detected_tags = {}
    st.session_state.completed_tags = {}
    st.session_state.current_label_index = 0
    st.session_state.labels_to_ask = []
    st.session_state.generated_questions = {}
    st.session_state.ragflow_response = ""
    st.session_state.ragflow_reference = ""
    st.session_state.session_id = ""
    st.session_state.greeting = ""
    st.session_state.processing = False  # Track if we're processing

# RAGFlow configuration
RAGFLOW_BASE = "http://localhost:80"
RAGFLOW_API_KEY = "ragflow-MxYTc0ZDhlOWY3ZDExZjBhOGFiN2FmNG"
RAGFLOW_CHAT_ID = "7cec5d2aa9b911f0922472143deb1554"

RAGFLOW_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RAGFLOW_API_KEY}",
}


def create_ragflow_session(chat_id: str, name: str = "reflection-session"):
    """Create a new session in RAGFlow"""
    url = f"{RAGFLOW_BASE}/api/v1/chats/{chat_id}/sessions"
    payload = {"name": name}
    resp = requests.post(url, headers=RAGFLOW_HEADERS, data=json.dumps(payload))

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Create session failed: HTTP {resp.status_code}, text={resp.text}")

    if resp.status_code != 200 or data.get("code") != 0 or "data" not in data:
        raise RuntimeError(f"Create session failed: {data}")

    session_id = data["data"]["id"]
    greeting = None
    msgs = data["data"].get("messages") or []
    if msgs and isinstance(msgs, list):
        first = msgs[0]
        if isinstance(first, dict) and first.get("role") == "assistant":
            greeting = first.get("content")

    return session_id, greeting


def query_ragflow(chat_id: str, session_id: str, question: str, stream: bool = False):
    """Query RAGFlow for a response"""
    url = f"{RAGFLOW_BASE}/api/v1/chats/{chat_id}/completions"
    payload = {
        "question": question,
        "stream": stream,
        "session_id": session_id
    }

    if not stream:
        resp = requests.post(url, headers=RAGFLOW_HEADERS, data=json.dumps(payload))
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Completion failed: HTTP {resp.status_code}, text={resp.text}")

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")

        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        elif "data" in data:
            d = data["data"]
            answer = d.get("answer") or d.get("message")
            if not answer and isinstance(d.get("messages"), list) and d["messages"]:
                answer = d["messages"][-1].get("content")
            reference = d.get("reference")
            return answer, reference
        else:
            msg = data.get("message") or data.get("answer")
            if msg:
                return msg, data.get("reference")
            raise RuntimeError(f"Unexpected response: {data}")
    return None, None


# Initialize LLM
@st.cache_resource
def load_llm():
    return Ollama(model="mistral")


llm = load_llm()


def get_no_example_prompt(style, form):
    """Create a prompt template without examples"""
    return PromptTemplate.from_template(f"""
You are a helpful assistant conducting a reflective conversation about an engineering project.
Please use the following style: {style}.

The user has already specified: {{tags}}.
Now generate **only one** short question to clarify the missing label: {{label}}.

Use this structure: "{form}"

Do NOT include any examples, explanations, or options.
Only return the final question text without using quotation marks, nothing else.
""")


def get_dynamic_question_chain(llm):
    """Create a dynamic question chain with random style and form"""
    QUESTION_STYLES = [
        "Be friendly and informal",
        "Use an academic tone",
        "Use a concise corporate tone",
        "Use a soft, empathetic style",
        "Use a humorous, light tone",
        "Ask as a curious colleague",
        "Ask as a retrospective facilitator"
    ]

    QUESTION_FORMS = [
        "What would you say is the most suitable {label}?",
        "Could you share your view on the {label}?",
        "Which {label} would best describe your case?",
        "In your opinion, what {label} fits this context?",
        "Help me understand what {label} applies best here.",
        "From your experience, which {label} matters most?"
    ]

    style = random.choice(QUESTION_STYLES)
    form = random.choice(QUESTION_FORMS)

    prompt = get_no_example_prompt(style, form)
    return LLMChain(llm=llm, prompt=prompt)


# Option definitions
OPTIONS = {
    "domains": ["product development", "engineering design"],
    "subdomains": [
        "requirement management and architecture development",
        "subsystem development and implementation",
        "system integration and protection",
        "project and risk management"
    ],
    "phases": ["feedback", "reflection", "planning"],
    "dimensions": ["social", "goal", "process"],
    "level": ["team level", "individual level"],
    "objectives": ["process adaption", "method adaption", "tool adaption", "adaption of results"]
}


# Load models and configuration
@st.cache_resource
def load_models():
    """Load the trained model and related files"""
    model = load_model("reflection_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_maps.pkl", "rb") as f:
        label_maps = pickle.load(f)
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    MAX_LEN = config['MAX_LEN']
    return model, tokenizer, label_maps, MAX_LEN


try:
    model, tokenizer, label_maps, MAX_LEN = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


def predict_with_model(user_input: str) -> dict:
    """Predict labels using the trained model"""
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    preds = model.predict(padded)

    result = {}
    label_order = ['domains', 'subdomains', 'phases', 'dimensions', 'level', 'objectives']
    for i, key in enumerate(label_order):
        pred_vector = preds[i][0]
        idx = np.argmax(pred_vector)
        label_list = sorted(label_maps[key].keys(), key=lambda x: label_maps[key][x])
        result[key] = label_list[idx]
    return result


keyword_exclude_map = {
    'phases': ['reflection']
}


def identify_from_input(user_input: str, options: dict) -> dict:
    """Identify labels from user input using both keyword matching and model prediction"""
    result = {}
    for key in options:
        matched = False
        for label in options[key]:
            if label.lower() in user_input.lower():
                if key in keyword_exclude_map and label.lower() in keyword_exclude_map[key]:
                    continue
                result[key] = label
                matched = True
                break
        if not matched:
            result[key] = 'none'
    model_result = predict_with_model(user_input)
    for key in result:
        if result[key] == 'none':
            result[key] = model_result[key]
    return result


def get_dynamic_question(label, tags_str):
    """Generate a dynamic question for the specified label"""
    question_chain = get_dynamic_question_chain(llm)
    response = question_chain.invoke({"tags": tags_str, "label": label})
    return response["text"].strip()


def generate_all_questions(labels_to_ask, completed_tags):
    """Generate all questions at once and store them in session state"""
    generated_questions = {}

    # Get the current tags string
    tags_str = ", ".join([f"{k}: {v}" for k, v in completed_tags.items() if v != "none"])

    # Generate questions for all labels to ask
    for label in labels_to_ask:
        question_text = get_dynamic_question(label, tags_str)
        generated_questions[label] = question_text

    return generated_questions


def build_ragflow_prompt(user_input: str, completed_tags: dict) -> str:
    """Build a prompt for RAGFlow based on user input and completed tags"""
    prompt = f"""Based on the following engineering project situation: "{user_input}"

And the following detailed reflection context:

- Domain: {completed_tags['domains']}
- Subdomain: {completed_tags['subdomains']}
- Reflection Phase: {completed_tags['phases']}
- Dimension: {completed_tags['dimensions']}
- Level: {completed_tags['level']}
- Objective: {completed_tags['objectives']}

Please generate 3-5 thoughtful, specific reflection questions that consider ALL of these contextual factors together and provide useful assistance for planning the reflection in the form of reflection questions. Please formulate general questions that help users analyze the strengths and weaknesses of the reflection example. The questions should:

1. Integrate multiple aspects of the context (not just focus on one label)
2. Be appropriate for the {completed_tags['phases']} phase of reflection
3. Address the {completed_tags['dimensions']} dimension at the {completed_tags['level']}
4. Help achieve the {completed_tags['objectives']} objective in {completed_tags['domains']}
5. Be specific to the subdomain of {completed_tags['subdomains']}

Generate questions that encourage deep thinking about how these different aspects interact in the described situation.
Please just return 3-5 precise and short generated questions, no other content is needed.
"""
    return prompt

def update_labels_to_ask(edit_mode: bool = False):
    """
    Generate the list of labels to ask the user.
    - If edit_mode is False: include only labels whose value is 'none' (except for phase rules).
    - If edit_mode is True: include all labels so the user can edit predictions.
    Phase-specific rule: if the current phase is 'feedback', dimensions and level are presented but
    their options will be limited to ['none'] in the UI (handled in step=1).
    """
    labels_to_ask = []
    for key in OPTIONS:
        # If edit mode, include label regardless of current value (subject to phase skipping rules below).
        if edit_mode:
            # Still skip showing dimensions/level entirely only if you really want to hide them;
            # here we include them so user can explicitly confirm 'none' when phase is feedback.
            labels_to_ask.append(key)
            continue

        # non-edit mode: only include labels that are not selected ('none')
        # but if phase is feedback, we do not include dimensions/level because they are logically 'none'
        if st.session_state.completed_tags.get("phases") == "feedback" and key in ["dimensions", "level"]:
            # If we want the user to still confirm 'none' in non-edit mode, change behavior;
            # currently keep them excluded in non-edit mode (initial flow).
            continue

        if st.session_state.completed_tags.get(key, "none") == "none":
            labels_to_ask.append(key)

    st.session_state.labels_to_ask = labels_to_ask

def reset_chat():
    """Reset the chat to initial state"""
    st.session_state.step = 0
    st.session_state.user_input = ""
    st.session_state.detected_tags = {}
    st.session_state.completed_tags = {}
    st.session_state.current_label_index = 0
    st.session_state.labels_to_ask = []
    st.session_state.generated_questions = {}
    st.session_state.ragflow_response = ""
    st.session_state.ragflow_reference = ""
    st.session_state.session_id = ""
    st.session_state.greeting = ""
    st.session_state.processing = False


# UI layout
st.title("🤔 Reflection Chatbot")
st.markdown("Welcome to the Reflection Chatbot! Please describe your situation for reflection.")

# 1. User input
if st.session_state.step == 0:
    # Use a form to prevent double submission
    with st.form(key="user_input_form"):
        user_input = st.text_area(
            "Describe your engineering project situation:",
            height=150,
            placeholder="Describe your project, challenges, or experiences that you'd like to reflect on..."
        )

        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        if user_input.strip():
            st.session_state.user_input = user_input
            st.session_state.processing = True
            st.session_state.step = 0.5  # Intermediate processing step
            st.rerun()
        else:
            st.warning("Please enter a description of your situation.")

# 1.5. Processing step (detect tags and generate questions)
elif st.session_state.step == 0.5:
    if st.session_state.processing:
        with st.spinner("Analyzing your input and preparing questions..."):
            # Detect tags from input
            st.session_state.detected_tags = identify_from_input(st.session_state.user_input, OPTIONS)
            st.session_state.completed_tags = st.session_state.detected_tags.copy()

            # Determine which labels need to be asked
            labels_to_ask = []
            for key in OPTIONS:
                if st.session_state.detected_tags[key] == 'none':
                    labels_to_ask.append(key)

            st.session_state.labels_to_ask = labels_to_ask

            # Generate all questions at once
            if labels_to_ask:
                st.session_state.generated_questions = generate_all_questions(
                    labels_to_ask, st.session_state.completed_tags
                )

            st.session_state.processing = False
            st.session_state.step = 1
            st.rerun()

# 2. Ask for missing labels
elif st.session_state.step == 1:
    st.subheader("Clarify Your Reflection Context")
    st.info("We need a bit more information to provide the most relevant reflection questions.")

    # If edit_mode flag is not set, default to False
    edit_mode = st.session_state.get("edit_mode", False)

    # Ensure labels_to_ask is updated; if not in edit_mode, keep normal behavior
    # (this call will keep labels_to_ask consistent if user navigates here directly)
    update_labels_to_ask(edit_mode=edit_mode)

    # Guard: if there are no labels to ask, go back to step 2
    if not st.session_state.labels_to_ask:
        st.session_state.step = 2
        # clear edit_mode if it was set
        st.session_state.edit_mode = False
        st.rerun()

    # Ensure current_label_index is within range
    if st.session_state.current_label_index >= len(st.session_state.labels_to_ask):
        st.session_state.current_label_index = 0

    # If there are still labels to select
    if st.session_state.current_label_index < len(st.session_state.labels_to_ask):
        current_label = st.session_state.labels_to_ask[st.session_state.current_label_index]

        # Display the generated question for the label if present
        if current_label in st.session_state.generated_questions:
            question_text = st.session_state.generated_questions[current_label]
            st.markdown(f"**{question_text}**")

        # Special hints for phase label
        if current_label == "phases":
            with st.expander("ℹ️ Hints for choosing phase"):
                st.markdown("""
                - **Feedback**: Review the current or past event and gather factual information related to the outcome or the process.
                - **Reflection**: Critically analyze the experience, identify problems and areas for improvement, and summarize actionable insights.
                - **Planning**: Develop a concrete action plan based on the reflection to improve future practices.
                """)

        # Determine options dynamically:
        # - If current label is 'dimensions' or 'level' AND the currently selected phase is 'feedback',
        #   then restrict the choices to ['none'].
        if current_label in ["dimensions", "level"] and st.session_state.completed_tags.get("phases") == "feedback":
            options = ["none"]
        else:
            options = OPTIONS[current_label]

        # Try to set the default selection index to the current completed tag value if present
        current_value = st.session_state.completed_tags.get(current_label, "none")
        try:
            default_index = options.index(current_value) if current_value in options else 0
        except Exception:
            default_index = 0

        # Radio with index so it defaults to the predicted/previous value
        selected_option = st.radio(
            f"Select {current_label}:",
            options,
            index=default_index,
            key=f"option_{current_label}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Selection", key=f"confirm_{current_label}"):
                # Save the user's selection (overwrites LSTM prediction if any)
                st.session_state.completed_tags[current_label] = selected_option

                # Advance to next label
                st.session_state.current_label_index += 1

                # If we were in edit_mode and we have finished editing all labels,
                # reset edit_mode to False so future flows behave normally.
                if st.session_state.edit_mode and st.session_state.current_label_index >= len(st.session_state.labels_to_ask):
                    st.session_state.edit_mode = False

                # Update labels_to_ask (respecting whether we are still in edit_mode)
                update_labels_to_ask(edit_mode=st.session_state.get("edit_mode", False))

                # If no more labels to ask, move to step 2
                if st.session_state.current_label_index >= len(st.session_state.labels_to_ask):
                    st.session_state.step = 2
                st.rerun()

        with col2:
            if st.button("Back to Description", key=f"back_{current_label}"):
                # If user wants to go back to description, fully reset
                reset_chat()
                st.rerun()

    # If all labels handled (safety), go to step 2
    else:
        st.session_state.step = 2
        st.rerun()


# 3. Show collected information and generate questions
elif st.session_state.step == 2:
    st.success("All context information collected!")

    # Show all collected tags
    st.subheader("Collected Context Information")
    context_cols = st.columns(2)
    with context_cols[0]:
        st.info(f"**Domain**: {st.session_state.completed_tags['domains']}")
        st.info(f"**Subdomain**: {st.session_state.completed_tags['subdomains']}")
        st.info(f"**Phase**: {st.session_state.completed_tags['phases']}")
    with context_cols[1]:
        st.info(f"**Dimension**: {st.session_state.completed_tags['dimensions']}")
        st.info(f"**Level**: {st.session_state.completed_tags['level']}")
        st.info(f"**Objective**: {st.session_state.completed_tags['objectives']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate Reflection Questions", type="primary"):
            with st.spinner("Generating reflection questions..."):
                ragflow_prompt = build_ragflow_prompt(st.session_state.user_input, st.session_state.completed_tags)

                try:
                    session_id, greeting = create_ragflow_session(RAGFLOW_CHAT_ID)
                    st.session_state.session_id = session_id
                    st.session_state.greeting = greeting

                    answer, reference = query_ragflow(RAGFLOW_CHAT_ID, session_id, ragflow_prompt, stream=False)

                    st.session_state.ragflow_response = answer
                    st.session_state.ragflow_reference = reference
                    st.session_state.step = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"Error querying RAGFlow: {e}")

    with col2:
        if st.button("Edit Context Labels", type="secondary"):
            # Enter edit mode so user can change even LSTM-predicted labels
            st.session_state.step = 1
            st.session_state.current_label_index = 0
            st.session_state.edit_mode = True  # set a flag to indicate editing
            update_labels_to_ask(edit_mode=True)
            st.rerun()

    with col3:
        if st.button("Back to Description", key="back_to_description"):
            reset_chat()
            st.rerun()

# 4. Show results
elif st.session_state.step == 3:
    st.subheader("Generated Reflection Questions")

    if st.session_state.ragflow_response:
        st.success(st.session_state.ragflow_response)
    else:
        st.warning("Sorry, no reflection questions could be generated.")

    if st.session_state.ragflow_reference:
        with st.expander("References"):
            st.write(st.session_state.ragflow_reference)

    if st.button("Start New Reflection", type="primary"):
        reset_chat()
        st.rerun()

# Sidebar information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This reflection chatbot helps engineering professionals 
    analyze their projects and experiences through guided reflection.

    It uses AI to:
    - Understand your project context
    - Identify relevant reflection dimensions
    - Generate thoughtful questions for deeper analysis
    """)

    if st.session_state.step > 0 and st.session_state.user_input:
        st.divider()
        st.subheader("Your Input")
        st.caption(st.session_state.user_input)

    st.divider()
    st.caption("Built with Streamlit, LangChain, and RAGFlow")

# Add some custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)
