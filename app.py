import streamlit as st
import os
import random
import io
import time
from groq import Groq
import requests
from urllib.parse import quote
from PIL import Image
import re

# Configure API key
GROQ_KEY = os.getenv("GROQ_KEY", st.secrets.get("GROQ_KEY", ""))

# Initialize Groq client
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

def validate_content(text, content_type):
    """
    Comprehensive content validation focusing on safety and appropriateness

    Args:
        text (str): The content to validate
        content_type (str): Type of content being validated (for error messages)

    Returns:
        tuple: (is_valid, message)
    """
    if not text.strip():
        return False, f"The {content_type} cannot be empty."

    if not GROQ_KEY:
        return False, f"API key is missing. Cannot validate {content_type}."

    try:
        # Content safety check
        safety_prompt = f"""You are a content moderation AI Validator. 
        TASK: Analyze the provided {text} and determine if it contains the following content.
        - Check for: hate speech, violence, harassment, illegal activities, sexual content, adulterated content
        - Also flag content promoting: self-harm, terrorism, scams, discrimination, abuse
        - Consider context carefully to avoid false positives
        
        OUTPUT FORMAT: Respond with ONLY ONE of these options:
        "SAFE" - If content is appropriate for use
        "UNSAFE: [reason]" - If content violates guidelines (briefly explain why)
        """

        safety_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": safety_prompt},
                {"role": "user", "content": f"Moderate this {content_type}: {text}"}
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.2,
            max_tokens=300
        )

        safety_result = safety_response.choices[0].message.content.strip()

        if safety_result.startswith("UNSAFE"):
            return False, safety_result.replace("UNSAFE:", "").strip()

        return True, "Content appears appropriate"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def generate_content(content_brief, platform, creativity):
    """Generate both image prompt and copy from a content brief with platform-specific length"""

    # Validate content brief
    is_valid, error_msg = validate_content(content_brief, content_type="content brief")
    if not is_valid:
        return "INAPPROPRIATE", error_msg, "Content moderation rejected the request"

    # Platform-specific character limits
    platform_limits = {
        "LinkedIn": "1200-1500",
        "Twitter": "350",
        "Instagram": "700-1000"
    }

    system_prompt = f"""You are an expert social media content creator for {platform}.

TASK: Generate TWO distinct outputs separated by [SEPARATOR] without any explanations or thinking:

1. SOCIAL MEDIA COPY:
- Create an engaging {platform}-optimized post with the right tone and emoji usage 
- Include relevant hashtags tailored specifically for {platform}'s audience
- Add a clear call-to-action that drives engagement
- Ensure the copy feels natural and human-written, not AI-generated
- IMPORTANT: Keep length within {platform_limits.get(platform, "700-1000")} characters for {platform}

2. IMAGE PROMPT:
- Create a detailed visual prompt for professional AI image generation
- Include specific subject positioning, camera angle, lighting, and mood
- Specify a cohesive color palette that aligns with the message tone
- Add depth cues like foreground/midground/background elements
- Include fine details like textures, materials, and environmental elements
- Specify photorealistic quality, 8k resolution, and professional photography

FORMAT:
[SOCIAL MEDIA COPY]
Your generated copy here...

[SEPARATOR]

[IMAGE PROMPT]
Your detailed image generation prompt here...
"""

    if not GROQ_KEY:
        return "ERROR", "API key is missing. Please configure your GROQ_KEY.", "Missing API key"

    try:
        # Generate content using LLM
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create engaging {platform} content for: {content_brief}"}
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=creativity,
            max_tokens=800
        )

        result = response.choices[0].message.content

        # Clean up any thinking traces
        for tag in ["<think>", "</think>", "<reasoning>", "</reasoning>", "<notes>", "</notes>"]:
            if tag in result:
                result = re.sub(f"{tag}.*?{tag.replace('<', '</') if not tag.startswith('</') else tag}", "", result, flags=re.DOTALL)

        # Split the output
        if "[SEPARATOR]" in result:
            parts = result.split("[SEPARATOR]")
            social_copy = parts[0].replace("[SOCIAL MEDIA COPY]", "").strip()
            image_prompt = parts[1].replace("[IMAGE PROMPT]", "").strip()
        else:
            # Fallback if separator not found
            social_copy = result
            image_prompt = f"Professional {platform} image showing: {content_brief}, 8k resolution, professional photography, detailed, cinematic lighting"

        # Check for refusal indicators
        refusal_indicators = [
            "cannot generate", "unable to create",  
            "violates guidelines", "against policy", "restricted", "not feasible"
        ]

        if any(indicator.lower() in social_copy.lower() for indicator in refusal_indicators) or \
                any(indicator.lower() in image_prompt.lower() for indicator in refusal_indicators):
            return "INAPPROPRIATE", "The AI declined to generate this content. Please adjust your request.", "Content moderation rejected the request"

        # Validate both outputs (without relevance context since this is initial generation)
        for content, content_type in [(social_copy, "social copy"), (image_prompt, "image prompt")]:
            is_valid, error_msg = validate_content(content, content_type=content_type)
            if not is_valid:
                return "INAPPROPRIATE", error_msg, "Content moderation rejected the request"

        return "SUCCESS", social_copy, image_prompt
    except Exception as e:
        return "ERROR", f"Failed to generate content: {str(e)}", f"Image about {content_brief}"

def generate_image(prompt, aspect_ratio):
    """Generate image using Pollinations.ai API"""

    # Validate the image prompt
    is_valid, error_msg = validate_content(prompt, content_type="image prompt")
    if not is_valid:
        raise Exception(error_msg)

    # Map aspect ratio to dimensions
    aspect_map = {
        "1:1": (1080, 1080),     # Square
        "4:5": (1080, 1350),     # Portrait
        "16:9": (1200, 675),     # Landscape
        "9:16": (1080, 1920),    # Story/Reel
        "2:3": (1080, 1620)      # Pinterest
    }
    width, height = aspect_map.get(aspect_ratio, (1080, 1080))

    # Enhance prompt if needed
    core_prompt = prompt.strip()
    if not any(term in core_prompt.lower() for term in ["8k", "resolution", "professional", "detailed"]):
        core_prompt += ", 8k resolution, professional photography, sharp focus, perfect composition, highly detailed"

    # URL-encode the prompt
    encoded_prompt = quote(core_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&nofeed=true&model=stable-diffusion&enhance=true&seed={random.randint(0, 1000000)}&width={width}&height={height}"

    # Attempt image download with retries
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Verify we got an actual image
            image_data = response.content
            try:
                Image.open(io.BytesIO(image_data))
                return image_data, core_prompt
            except Exception:
                if attempt == max_retries - 1:
                    raise Exception("Received invalid image data")
                time.sleep(retry_delay)
                retry_delay += 2
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Image generation failed after {max_retries} attempts: {str(e)}")
            time.sleep(retry_delay)
            retry_delay += 2

    raise Exception("Failed to generate image after multiple attempts")

def refine_content(content_type, feedback, platform):
    """Refine either image prompt or social copy based on feedback using LLM"""

    # Validate the feedback and check relevance to original content
    is_valid, error_msg = validate_content(feedback, content_type="feedback")
    if not is_valid:
        return "INAPPROPRIATE", error_msg

    if not GROQ_KEY:
        return "ERROR", "API key is missing. Please configure your GROQ_KEY."

    # Retrieve existing content based on content_type
    if content_type == "image":
        existing_content = st.session_state.image_prompt
        system_prompt = f"""You are a professional AI image prompt engineer.
TASK: Produce a refined image prompt based on existing prompt and user feedback.
- Existing Prompt: {existing_content}
- Integrate the user's feedback while creating a new compelling image concept
- Maintain core elements of the original prompt
- Ensure detailed specifications for subject, lighting, colors, perspective, and environment
- Always include technical specs: 8k resolution, professional photography, ultra detailed
- NEVER create prompts for inappropriate, offensive, violent, explicit, or unsafe imagery
- Focus on clean, professional imagery suitable for business use



USER FEEDBACK: {feedback}

OUTPUT FORMAT: Provide the new refined image prompt only."""
    else:  # For social copy
        existing_content = st.session_state.social_copy
        platform_limits = {
            "LinkedIn": "1200-1500",
            "Twitter": "240",
            "Instagram": "700-1000"
        }

        system_prompt = f"""You are a professional social media content specialist for {platform}.
TASK: Produce refined social media copy based on existing content and user feedback.
- Existing Copy: {existing_content}
- Create new engaging {platform}-optimized post that incorporates {feedback}
- Maintain the core message and intent of the original copy 
- Follow {platform} best practices for engagement
- Include relevant hashtags tailored specifically for {platform}'s audience
- Add a clear call-to-action that drives engagement
- IMPORTANT: Keep length within {platform_limits.get(platform, "700-1000")} characters for {platform}
- Include relevant hashtags tailored specifically for {platform}'s audience



USER FEEDBACK: {feedback}

OUTPUT FORMAT: Provide the new refined copy only.
If feedback is irrelevant, explain why it cannot be used for copy refinement."""

    try:
        # Use Groq to refine the content
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Refine the {content_type} with this feedback, considering the existing content"}
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            max_tokens=800
        )

        result = response.choices[0].message.content.strip()

        

        # Clean up any thinking traces
        for tag in ["<think>", "</think>", "<reasoning>", "</reasoning>", "<notes>", "</notes>"]:
            if tag in result:
                result = re.sub(f"{tag}.*?{tag.replace('<', '</') if not tag.startswith('</') else tag}", "", result, flags=re.DOTALL)

        # Validate refined content (without context check since we've already verified relevance)
        is_valid, error_msg = validate_content(result,  content_type=f"refined {content_type}")
        if not is_valid:
            return "INAPPROPRIATE", error_msg

        return "SUCCESS", result

    except Exception as e:
        return "ERROR", f"Refinement failed: {str(e)}"

def run_app():
    # Streamlit UI Configuration
    st.set_page_config(
        page_title="Social Media Creator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS styles
    st.markdown("""
    <style>
        .main .block-container {padding-top: 2rem;}
        .stButton>button {
            height: 3rem; font-size: 1.1rem; background-color: #4F8BF9;
            color: white; border-radius: 5px; border: none;
        }
        .card {padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: white; margin-bottom: 1rem;}
        .copy-display {background-color: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #e0e0e0; margin-top: 1rem;}
        .error-message {background-color: #ffebee; border-left: 4px solid #f44336; color: #d32f2f; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.5rem 0;}
        .success-message {background-color: #e8f5e9; border-left:4px solid #4caf50; color: #2e7d32; padding: 0.5rem 1rem; border-radius: 4px; margin: 0.5rem 0;}
        .character-count {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: right; font-weight: bold;}
        .gradient-header {background: linear-gradient(to right, #1a237e, #3949ab, #42a5f5); padding: 1rem; border-radius: 8px; color: white !important; text-align: center; margin-bottom: 1.5rem;}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    state_vars = {
        'social_copy': "", 'image_data': None, 'image_prompt': "",
        'generation_complete': False, 'conversation_history': [], 'input_error': ""
    }

    for key, default in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Validate API key
    api_error = "" if GROQ_KEY else "⚠️ API Key Missing: Please set your GROQ_KEY in the environment variables or secrets."
    if api_error:
        st.session_state.input_error = api_error

    # Main Interface
    st.markdown("<h1 class='gradient-header'>🚀 Social Media Post Generator</h1>", unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Platform Settings")

        # Platform options
        platforms = ["LinkedIn", "Twitter", "Instagram"]
        platform = st.selectbox("Select a Social Platform", options=platforms)

        # Aspect ratio selection
        aspect_ratio_options = {
            "1:1": "Square (universal)",
            "4:5": "Portrait (Instagram)",
            "16:9": "Landscape (LinkedIn)",
            "9:16": "Story/Reel (vertical)",
            "2:3": "Pinterest (tall)"
        }

        aspect_ratio = st.selectbox(
            "Image Aspect Ratio",
            list(aspect_ratio_options.keys()),
            index=0,
            format_func=lambda x: f"{x} - {aspect_ratio_options[x]}"
        )

        st.divider()

        st.markdown("### 📝 Content Settings")
        creativity = st.slider(
            "Creativity Level",
            min_value=0.2, max_value=1.0, value=0.7,
            help="Higher values create more varied content, lower values are more consistent"
        )

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Create Content", "Content History"])

    with tab1:
        # Input area
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Content Brief")

        content_brief = st.text_area(
            "Describe what you want to post about",
            "A modern workspace with plants and natural light demonstrating how our company promotes employee wellbeing and productivity through thoughtful office design.",
            height=100
        )

        # Display input error if present
        if st.session_state.input_error:
            st.markdown(f"<div class='error-message'>{st.session_state.input_error}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Generation Section
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            generate_button = st.button(f"✨ Create {platform} Post", use_container_width=True)
        with col_btn2:
            reset_button = st.button("🔄 Reset", use_container_width=True)

        if reset_button:
            for key in ['social_copy', 'image_data', 'image_prompt', 'input_error']:
                st.session_state[key] = "" if key in ['social_copy', 'image_prompt', 'input_error'] else None
            st.session_state.generation_complete = False
            st.rerun()

        if generate_button:
            if api_error:
                st.session_state.input_error = api_error
                st.rerun()
            elif not content_brief.strip():
                st.session_state.input_error = "⚠️ Please enter a content brief before generating."
                st.rerun()
            else:
                try:
                    st.session_state.input_error = ""  # Clear any previous errors

                    # Generate both copy and image prompt
                    with st.spinner("Creating your professional post..."):
                        result_status, social_copy, image_prompt = generate_content(content_brief, platform, creativity)

                        if result_status == "ERROR":
                            st.session_state.input_error = f"⚠️ {social_copy}"
                            st.rerun()

                        if result_status == "INAPPROPRIATE":
                            st.session_state.input_error = "⚠️ Cannot generate this content: The request appears to violate our content guidelines."
                            st.rerun()

                        image_data, enhanced_prompt = generate_image(image_prompt, aspect_ratio)

                        # Store in session state
                        st.session_state.social_copy = social_copy
                        st.session_state.image_data = image_data
                        st.session_state.image_prompt = enhanced_prompt
                        st.session_state.generation_complete = True

                        # Add to conversation history
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.conversation_history.append({
                            "platform": platform,
                            "content_brief": content_brief,
                            "social_copy": social_copy,
                            "image_data": image_data,
                            "image_prompt": enhanced_prompt,
                            "timestamp": timestamp
                        })

                except Exception as e:
                    st.session_state.input_error = f"⚠️ Generation failed: {str(e)}"
                    st.rerun()

        # Display Results if generation is complete
        if st.session_state.generation_complete and st.session_state.image_data is not None:
            # Side-by-side layout for image and text
            col1, col2 = st.columns(2)

            # Image column
            with col1:
                st.markdown("<div class='card side-container'>", unsafe_allow_html=True)
                st.subheader("Visual Preview")
                st.image(st.session_state.image_data, use_container_width=True)

                st.download_button(
                    "📥 Download Image",
                    st.session_state.image_data,
                    file_name=f"{platform.lower()}_post_{int(time.time())}.png",
                    mime="image/png"
                )

                # Image refinement
                st.markdown("### 🔄 Refine Image")
                image_feedback = st.text_area(
                    "How would you like to improve the image?",
                    placeholder="e.g., 'Make colors more vibrant', 'Change to outdoor setting'",
                    key="image_feedback",
                    height=80
                )

                image_feedback_error = st.empty()

                if st.button("Update Image", key="update_image_btn", use_container_width=True):
                    if image_feedback:
                        with st.spinner("Generating improved image..."):
                            if api_error:
                                image_feedback_error.markdown(f"<div class='error-message'>{api_error}</div>", unsafe_allow_html=True)
                            else:
                                status, result = refine_content("image", image_feedback, platform)

                                if status == "SUCCESS":
                                    try:
                                        new_image_data, updated_prompt = generate_image(result, aspect_ratio)
                                        st.session_state.image_data = new_image_data
                                        st.session_state.image_prompt = updated_prompt

                                        # Update conversation history
                                        if st.session_state.conversation_history:
                                            st.session_state.conversation_history[-1]["image_data"] = new_image_data
                                            st.session_state.conversation_history[-1]["image_prompt"] = updated_prompt

                                        image_feedback_error.markdown("<div class='success-message'>Image updated successfully!</div>", unsafe_allow_html=True)
                                        st.rerun()
                                    except Exception as e:
                                        image_feedback_error.markdown(f"<div class='error-message'>Image generation failed: {str(e)}</div>", unsafe_allow_html=True)
                                else:
                                    image_feedback_error.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)
                    else:
                        image_feedback_error.markdown("<div class='error-message'>Please provide specific feedback on how to improve the image.</div>", unsafe_allow_html=True)

                # Image prompt in expandable section
                with st.expander("View Image Prompt"):
                    st.code(st.session_state.image_prompt, language="text")

                st.markdown("</div>", unsafe_allow_html=True)

            # Text column
            with col2:
                st.markdown("<div class='card side-container'>", unsafe_allow_html=True)
                st.subheader(f"{platform} Copy")

                # Character count display
                char_count = len(st.session_state.social_copy)
                char_limits = {"LinkedIn": 3000, "Twitter": 280, "Instagram": 2200}

                limit_status = "within limit" if char_count <= char_limits[platform] else "exceeds limit"
                limit_color = "#2e7d32" if char_count <= char_limits[platform] else "#d32f2f"

                st.markdown(f"""
                <div class="character-count" style="color: {limit_color}">
                    {char_count} / {char_limits[platform]} characters ({limit_status})
                </div>
                """, unsafe_allow_html=True)

                # Display the copy
                st.markdown(f"<div class='copy-display'>{st.session_state.social_copy}</div>", unsafe_allow_html=True)

                st.download_button(
                    "📥 Download Copy",
                    st.session_state.social_copy,
                    file_name=f"{platform.lower()}_copy_{int(time.time())}.txt",
                    mime="text/plain"
                )

                # Text refinement
                st.markdown("### 🔄 Refine Copy")
                text_feedback = st.text_area(
                    "How would you like to improve the text?",
                    placeholder=f"e.g., 'Make tone more professional', 'Shorten for {platform}'",
                    key="text_feedback",
                    height=80
                )

                text_feedback_error = st.empty()

                if st.button("Update Copy", key="update_copy_btn", use_container_width=True):
                    if text_feedback:
                        with st.spinner("Updating copy..."):
                            if api_error:
                                text_feedback_error.markdown(f"<div class='error-message'>{api_error}</div>", unsafe_allow_html=True)
                            else:
                                status, result = refine_content("text", text_feedback, platform)

                                if status == "SUCCESS":
                                     # Clean up any thinking traces
                                    for tag in ["<think>", "</think>", "<reasoning>", "</reasoning>", "<notes>", "</notes>"]:
                                        if tag in result:
                                            result = re.sub(f"{tag}.*?{tag.replace('<', '</') if not tag.startswith('</') else tag}", "", result, flags=re.DOTALL)

                                    st.session_state.social_copy = result

                                    # Update conversation history
                                    if st.session_state.conversation_history:
                                        st.session_state.conversation_history[-1]["social_copy"] = result

                                    text_feedback_error.markdown("<div class='success-message'>Copy updated successfully!</div>", unsafe_allow_html=True)
                                    st.rerun()
                                else:
                                    text_feedback_error.markdown(f"<div class='error-message'>{result}</div>", unsafe_allow_html=True)
                    else:
                        text_feedback_error.markdown("<div class='error-message'>Please provide specific feedback on how to improve the copy.</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # History tab
    with tab2:
        if not st.session_state.conversation_history:
            st.info("📝 Your content history will appear here after you create your first post.")
        else:
            st.markdown("### 📚 Your Content History")

            # Filter history by platform
            filter_platform = st.multiselect(
                "Filter by platform",
                ["LinkedIn", "Twitter", "Instagram"],
                default=["LinkedIn", "Twitter", "Instagram"]
            )

            # Display historical content - reversed (newest first)
            for i, item in enumerate(sorted(
                [item for item in st.session_state.conversation_history if item["platform"] in filter_platform],
                key=lambda x: x["timestamp"],
                reverse=True
            )):
                with st.expander(f"**{item['platform']}** - {item['timestamp']} - {item['content_brief'][:50]}..."):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Visual")
                        st.image(item["image_data"], use_container_width=True)
                        st.download_button(
                            "📥 Download Image",
                            item["image_data"],
                            file_name=f"{item['platform'].lower()}_post_{i}.png",
                            mime="image/png",
                            key=f"dl_img_{i}"
                        )

                    with col2:
                        st.markdown(f"#### {item['platform']} Copy")
                        st.markdown(f"<div class='copy-display'>{item['social_copy']}</div>", unsafe_allow_html=True)
                        st.download_button(
                            "📥 Download Copy",
                            item["social_copy"],
                            file_name=f"{item['platform'].lower()}_copy_{i}.txt",
                            mime="text/plain",
                            key=f"dl_copy_{i}"
                        )

if __name__ == "__main__":
    run_app()