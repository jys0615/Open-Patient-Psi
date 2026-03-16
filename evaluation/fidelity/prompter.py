def generate_fidelity_prompt(conversation: str, response: str, dimension: str) -> str:
    return (
        f"You are evaluating the realism of a simulated patient.\n"
        f"Conversation:\n{conversation}\n\n"
        f"Response:\n{response}\n\n"
        f"On a scale of 1 to 5, rate the fidelity of the response in terms of {dimension}.\n"
        f"Answer with a single number only."
    )


def generate_accuracy_prompt(conversation: str, response: str, component_type: str, options: list[str]) -> str:
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    return (
        f"You are identifying the correct {component_type} reflected in the patient's response.\n"
        f"Conversation:\n{conversation}\n\n"
        f"Response:\n{response}\n\n"
        f"Which of the following best represents the {component_type}?\n"
        f"{options_text}\n\n"
        f"Answer with the number of the best option."
    )