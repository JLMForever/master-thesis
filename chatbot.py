import pickle
import random
import yaml
import numpy as np
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from keras.models import load_model
from keras.utils import pad_sequences

# 1. Initialization
llm = Ollama(model="mistral")

# 2. Build question templates
# Avoid questions that include inappropriate examples
def get_no_example_prompt(style, form):
    return PromptTemplate.from_template(f"""
You are a helpful assistant conducting a reflective conversation about an engineering project.
Please use the following style: {style}.

The user has already specified: {{tags}}.
Now generate **only one** short question to clarify the missing label: {{label}}.

Use this structure: "{form}"

Do NOT include any examples, explanations, or options.
Only return the final question text without using quotation marks, nothing else.
""")

# Dynamically generate questions in different formats to mimic human conversation
def get_dynamic_question_chain(llm):
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

# 3. Labels
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

# 4. Identification of labels
# Load LSTM model and artifacts
model = load_model("reflection_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_maps.pkl", "rb") as f:
    label_maps = pickle.load(f)
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

MAX_LEN = config['MAX_LEN']

def predict_with_model(user_input: str) -> dict:
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
    # supplement with model prediction
    model_result = predict_with_model(user_input)
    for key in result:
        if result[key] == 'none':
            result[key] = model_result[key]
    return result

# 5. Dynamically generate questions and validate user input
def ask_choice(label, choices, result, llm):
    tags_str = ", ".join([f"{k}: {v}" for k, v in result.items() if v != "none"])
    question_chain = get_dynamic_question_chain(llm)
    response = question_chain.invoke({"tags": tags_str, "label": label})
    question_text = response["text"].strip()

    print(f"\n{question_text}")
    print(f"(Please choose one of the following: {choices})")

    if label == "phase":
        print("\n（Hints for choosing phase:")
        print("- feedback: Review the current or past event and gather factual information related to the outcome or the process.")
        print("- reflection: Critically analyze the experience, identify problems and areas for improvement, and summarize actionable insights.")
        print("- planning: Develop a concrete action plan based on the reflection to improve future practices.）")

    while True:
        ans = input("> ").strip().lower()
        if ans in choices:
            return ans
        print(f"Invalid {label}. Please choose from: {choices}")

# 6. Multiple rounds of questioning
def prompt_missing_tags(detected: dict, options: dict, question_chain) -> dict:
    result = dict(detected)

    if result["domains"] == "none":
        result["domains"] = ask_choice("domain", options["domains"], result, question_chain)

    if result["subdomains"] == "none":
        result["subdomains"] = ask_choice("subdomain", options["subdomains"], result, question_chain)

    if result["phases"] == "none":
        result["phases"] = ask_choice("phase", options["phases"], result, question_chain)

    if result["phases"] == "feedback":
        result["dimensions"] = "none"
        result["level"] = "none"
    else:
        if result["dimensions"] == "none":
            result["dimensions"] = ask_choice("dimension", options["dimensions"], result, question_chain)
        if result["level"] == "none":
            result["level"] = ask_choice("level", options["level"], result, question_chain)

    if result["objectives"] == "none":
        result["objectives"] = ask_choice("objective", options["objectives"], result, question_chain)

    return result

# 7. Main chat function
def chat():
    print("\nWelcome to the Reflection Chatbot!")
    print("Please describe your situation for reflection:")

    user_input = input("\nYour input:\n> ")

    # Step 1: identify labels
    detected = identify_from_input(user_input, OPTIONS)

    # Step 2: multi-rounds questioning
    completed = prompt_missing_tags(detected, OPTIONS, llm)

    # Step 3: return reflection questions
    with open("data sample.yml", "r", encoding="utf-8") as f:
        samples = yaml.safe_load(f)

    def match_sample_questions(completed):
        for entry in samples:
            if (entry['domains'][0] == completed['domains'] and
                entry['subdomains'][0] == completed['subdomains'] and
                entry['phases'][0] == completed['phases'] and
                entry['dimensions'][0] == completed['dimensions'] and
                entry.get('level', ['none'])[0] == completed['level'] and
                entry['objectives'][0] == completed['objectives']):
                return entry['questions']
        return None

    questions = match_sample_questions(completed)

    # Step 4: output result
    print("\nAccording to your input, the corresponding reflection questions are:\n")
    if questions:
        for q in questions:
            print(f"- {q}")
    else:
        print("Sorry, no matching questions found in the sample database.")

    # Step 5: next round
    again = input("\nWould you like to reflect on another situation? (y/n): ").strip().lower()
    if again == 'y':
        chat()
    else:
        print("\nGoodbye. Thank you for using this Chatbot!")

# === Run ===
if __name__ == "__main__":
    chat()
