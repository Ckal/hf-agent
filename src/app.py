"""Modified Gradio UI integration to fix type errors."""
# pylint: disable=line-too-long, broad-exception-caught
import datetime
import pytz
import yaml
import re
from loguru import logger
from smolagents import (
    CodeAgent,
    HfApiModel,
    load_tool,
    tool,
)
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool


# Custom tools
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """
    Run a tool that does nothing yet.
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return f"What magic will you build? {arg1=}, {arg2=}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    Run a tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    logger.debug(f"get_current_time_in_timezone {timezone=}")
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        logger.error(e)
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


# Define our default prompt templates with the correct structure
DEFAULT_PROMPTS = {
    "system_prompt": "You are an expert assistant who can solve any task using code blobs.",
    "final_answer": {
        "pre_messages": "Here is the final answer to your question:",
        "post_messages": "Hope this helps!"
    },
    "planning": {
        "initial_facts": "First, let's identify what we know and what we need to find out.",
        "initial_plan": "Let's develop a step-by-step plan to solve this task.",
        "update_facts_pre_messages": "Let's update what we know based on our progress.",
        "update_facts_post_messages": "Here's our updated knowledge about the task.",
        "update_plan_pre_messages": "Let's review our plan based on what we've learned.",
        "update_plan_post_messages": "Here's our revised plan to complete the task."
    },
    # Adding the managed_agent section that was missing
    "managed_agent": {
        "task": "You're a helpful agent named '{{name}}'. You have been submitted this task by your manager.",
        "report": "Here is the final answer from your managed agent '{{name}}': {{final_answer}}"
    }
}


def create_agent():
    """Create and configure the agent with proper error handling."""
    # Initialize tools
    final_answer = FinalAnswerTool()
    visit_webpage = VisitWebpageTool()
    
    try:
        model = HfApiModel()
        logger.info("Successfully loaded model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to initialize model: {e}")
    
    # Import tool from Hub with proper error handling
    try:
        image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
        logger.info("Successfully loaded image generation tool")
    except Exception as e:
        logger.error(f"Error loading image generation tool: {e}")
        image_generation_tool = None
    
    # Load prompt templates or use defaults
    prompt_templates = DEFAULT_PROMPTS.copy()
    try:
        with open("prompts.yaml", "r", encoding="utf8") as stream:
            loaded_prompts = yaml.safe_load(stream)
            if loaded_prompts:
                # Update system prompt if available
                if "system_prompt" in loaded_prompts:
                    prompt_templates["system_prompt"] = loaded_prompts["system_prompt"]
                
                # Handle final_answer format
                if "final_answer" in loaded_prompts:
                    if isinstance(loaded_prompts["final_answer"], dict):
                        if "pre_messages" in loaded_prompts["final_answer"]:
                            prompt_templates["final_answer"]["pre_messages"] = loaded_prompts["final_answer"]["pre_messages"]
                        if "post_messages" in loaded_prompts["final_answer"]:
                            prompt_templates["final_answer"]["post_messages"] = loaded_prompts["final_answer"]["post_messages"]
                    else:
                        # String format, use as pre_messages
                        prompt_templates["final_answer"]["pre_messages"] = loaded_prompts["final_answer"]
                
                # Handle planning section
                if "planning" in loaded_prompts and isinstance(loaded_prompts["planning"], dict):
                    for key in prompt_templates["planning"]:
                        if key in loaded_prompts["planning"]:
                            prompt_templates["planning"][key] = loaded_prompts["planning"][key]
                
                logger.info("Successfully merged prompt templates")
    except Exception as e:
        logger.error(f"Error loading prompts.yaml: {e}")
        logger.info("Using default templates")
    
    # Prepare tools list
    tools = [
        my_custom_tool,
        get_current_time_in_timezone,
        final_answer,
        visit_webpage,
    ]
    
    if image_generation_tool:
        tools.append(image_generation_tool)
    
    # Create and return the agent
    return CodeAgent(
        model=model,
        tools=tools,
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name="Agent",
        description="A CodeAgent with improved capabilities",
        prompt_templates=prompt_templates,
    )


def run_app():
    """
    Run the application with custom Gradio interface to avoid schema validation issues.
    This circumvents the error in json_schema_to_python_type.
    """
    # Import gradio only when needed to customize the interface
    try:
        import gradio as gr
        from Gradio_UI import GradioUI
    except ImportError as e:
        logger.error(f"Error importing Gradio: {e}")
        raise
    
    try:
        # Create the agent
        agent = create_agent()
        
        # Basic interface without relying on complex schema validation
        with gr.Blocks() as demo:
            gr.Markdown("# Code Agent Interface")
            
            with gr.Row():
                with gr.Column(scale=4):
                    input_text = gr.Textbox(
                        label="Your task",
                        placeholder="Describe your task here...",
                        lines=5
                    )
                    
                    submit_btn = gr.Button("Submit")
                    
                with gr.Column(scale=6):
                    output_text = gr.Markdown(label="Agent Response")
            
            # Handle submissions manually to avoid schema issues
            def process_request(task):
                try:
                    result = agent(task)
                    return result
                except Exception as e:
                    logger.error(f"Error in agent processing: {e}")
                    return f"Error processing your request: {str(e)}"
            
            submit_btn.click(
                fn=process_request,
                inputs=input_text,
                outputs=output_text
            )
        
        # Launch with minimal options
        demo.launch(
            server_name="0.0.0.0",
            share=False,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Critical error: {e}")


if __name__ == "__main__":
    run_app()