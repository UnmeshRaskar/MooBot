import streamlit as st
from openai import OpenAI
import pandas as pd
import re
import os
from PIL import Image

# Cow_id images folder
image_folder = "./cow_images_optimized" 

# Initialize OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# App title
st.set_page_config(page_title="MooBot Chatbot")

# st.title("🐄 MooBot: Dairy Farm Chatbot")
# st.write("Ask me anything about your cattle!")
st.markdown("<h1 style='font-size: 80px;'>🐄 MooBot: Dairy Farm Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 40px;'>Ask me anything about your cattle!</p>", unsafe_allow_html=True)


# Load the master schema CSV
try:
    master_schema = pd.read_csv("combined_intersection_df.csv")
    combined_intersection_df = master_schema  # dataframe name used in system prompt
except Exception as e:
    st.write(f"Error loading master schema: {e}")
    combined_intersection_df = pd.DataFrame()  # Empty dataframe as fallback

# System prompts for different query types
data_system_prompt = """You are an expert in Python and Pandas. Generate valid and efficient Pandas code to extract information from a dataframe called `combined_intersection_df` using the user's query.

The dataframe has:
- `timestamp`: A timestamp column.
- `C01_behavior` to `C16_behavior`: Behavior labels (1 to 7) for 16 cows.
- `C01_x`, `C01_y`, `C01_z` to `C16_x`, `C16_y`, `C16_z`: Location coordinates for 16 cows.

Behavior labels are defined as:
0: Unknown — The cow is not visible in any camera view.
1: Walking — Moving from one location to another between consecutive frames.
2: Standing — Legs are straight up, head is not at the feeding area.
3: Feeding head up — Head is at the feeding area, mouth is above the food.
4: Feeding head down — Head is at the feeding area, mouth touches the food.
5: Licking — Licking the mineral (salt) block.
6: Drinking — Drinking at a water trough, mouth touches the water.
7: Lying — Cow is lying in the stall.

If the user refers to behaviors by name (such as 'standing', 'feeding', or 'lying'), map them to the correct behavior codes.

Location regions are defined as:
- Resting Area: _x coordinates between -400 to 600, _y coordinates between -200 to 200.
- Feeding Area: _x coordinates between -1000 to 1000, _y coordinates between -650 to -500.
- Water troughs Area: _x coordinates between 1000 to 1100, _y coordinates between -200 to 200 or _x coordinates between -1100 to -800, _y coordinates between -200 to 200.
- Ventilation Area: _x coordinates between -1100 to -800, _y coordinates between -200 to 200.
- Common Barn Area: Locations not specified above.

If the user refers to any of these location regions by name (such as 'resting area', 'feeding area', or 'water troughs'), map them to the correct location coordinates.

IMPORTANT FORMATTING RULES:
1. Your response must contain ONLY Python code within a single ```python code block.
2. Do not include any commentary, markdown formatting, or explanations outside the code block.
3. Use standard ASCII characters for operators (use >= instead of ≥, <= instead of ≤).
4. Use Python comments (#) for any explanations within the code.
5. Ensure proper indentation and no line breaks within logical statements.
6. Do not use bold text or other markdown formatting inside the code.
7. Make sure your code is complete and can be executed as-is.

**Guidelines for Generating Code:**
1. Ensure `timestamp` is treated as a datetime using `pd.to_datetime()`.
2. Use `.between()` for coordinate range filtering instead of `.apply(lambda x: x.between())`, which is inefficient.
3. Perform time filtering using `.dt.hour` for timestamps.
4. Filter cow behavior using direct column-wise comparisons without `.any(axis=1)` unless necessary.
5. For checking multiple columns (e.g., behavior or location), iterate using a loop instead of applying unnecessary DataFrame-wide operations.
6. Provide clean, efficient, and readable code with necessary comments.

**Example:**
User Query: 'Get me all the cows who were standing and located between x = 100 to 200 and y = -200 to 200 from 2 PM to 5 PM.'
Expected Code:
```python
# Ensure timestamp is in datetime format
combined_intersection_df['timestamp'] = pd.to_datetime(combined_intersection_df['timestamp'])

# Filter for the specified time range (2 PM to 5 PM)
time_filtered_df = combined_intersection_df[
    (combined_intersection_df['timestamp'].dt.hour >= 14) & 
    (combined_intersection_df['timestamp'].dt.hour <= 17)
]

# Identify the cows matching the behavior and location criteria
cows = []
for i in range(1, 17):
    behavior_col = f'C{i:02d}_behavior'
    x_col = f'C{i:02d}_x'
    y_col = f'C{i:02d}_y'
    
    mask = (time_filtered_df[behavior_col] == 2) & \
        (time_filtered_df[x_col].between(100, 200)) & \
        (time_filtered_df[y_col].between(-200, 200))
    
    if mask.any():
        cows.append(f'C{i:02d}')

print(cows) 
```"""

info_system_prompt = """You are MooBot, an expert assistant for dairy farmers.
Answer the user's question about cattle health, management, or system capabilities.

About MooBot:
- MooBot is a specialized chatbot for dairy farm monitoring
- MooBot can analyze behavioral data for up to 16 cows in the herd
- MooBot tracks behaviors like walking, standing, feeding, licking, drinking, and lying
- MooBot monitors cow locations in different barn areas (resting, feeding, water troughs, ventilation)

Cattle Health Information:
- Heat stress occurs when temperature-humidity index exceeds 72
- Signs of heat stress include reduced feed intake, increased water consumption, and decreased milk production
- Healthy cows typically spend 3-5 hours per day eating and 12-14 hours lying down
- Reduced lying time can indicate health issues or environmental problems
- Cows normally drink 30-50 gallons of water per day
- A sudden change in behavior patterns may indicate illness or distress

Respond in a helpful, informative manner. If you're unsure, say so rather than making up information."""

conversation_system_prompt = """You are MooBot, a friendly assistant for dairy farmers.
Engage in natural conversation with the user. You're knowledgeable about cattle and dairy farming,
but your primary function is to help farmers monitor their cows' behavior and location data.

Keep responses friendly, concise, and helpful. Use a warm, personable tone that reflects your purpose as a helpful tool for farmers who care about their animals' wellbeing.

You can tell users that you can:
1. Track cow behaviors including walking, standing, feeding, licking, drinking, and lying
2. Monitor cow locations throughout different areas of the barn
3. Answer questions about cattle health and management
4. Analyze patterns in cow behavior data

If asked about specific data analysis, encourage users to phrase their question as a data query about cow behaviors or locations."""

classification_system_prompt = """Classify the user query into one of these categories:
1. 'data_query': User is asking for specific data from the cow dataset. Examples: "Show me cows that were standing between 2-5pm", "Which cows spent the most time in the feeding area?", "How many cows were lying down at noon?", "Find cows that switched from feeding to drinking within an hour"

2. 'info_query': User is asking about cattle information or system capabilities. Examples: "What is heat stress in cows?", "How much water should a cow drink per day?", "What can you tell me about cow behavior?", "What can MooBot do?", "Tell me about yourself"

3. 'conversation': General chat or simple greetings. Examples: "Hello", "How are you?", "Thanks for the help", "How are my cows doing today?", "Good morning"

Respond with just the category name."""

# Chat message storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["sender"]).write(msg["text"])
    if "images" in msg and msg["images"]:
        cols = st.columns(min(4, len(msg["images"])))
        for i, (cow_id, img_path) in enumerate(msg["images"].items()):
            col_idx = i % 4
            with cols[col_idx]:
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    st.image(image, caption=f"{cow_id}", use_container_width=True)
                else:
                    st.write(f"{cow_id}: Image not found")

def classify_query(query):
    """Determine if the query is data-related, informational, or conversational."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": classification_system_prompt},
                {"role": "user", "content": query}
            ]
        )
        classification = completion.choices[0].message.content.strip().lower()
        if "data" in classification:
            return "data_query"
        elif "info" in classification:
            return "info_query"
        else:
            return "conversation"
    except Exception as e:
        st.write(f"Error classifying query: {e}")
        return "data_query"  # Default to data query on error

def process_data_query(query):
    """Handle data queries using code generation."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": data_system_prompt},
                {"role": "user", "content": query}
            ]
        )

        # Get the generated code from OpenAI response
        generated_code = completion.choices[0].message.content
        # st.write(f"Generated Code: {generated_code}")  # Debugging line

        # Extract only the Python code part
        python_code = re.search(r'```python\n(.*?)```', generated_code, re.DOTALL)
        if python_code:
            python_code = python_code.group(1)
        else:
            return {"text": "Error: No Python code found in the response.", "images": {}}

        # Execute the code
        local_vars = {"combined_intersection_df": combined_intersection_df, "pd": pd}
        try:
            exec(python_code, globals(), local_vars)
            
            # If cows list is generated, prepare results
            if 'cows' in local_vars:
                cows = local_vars['cows']
                response_text = f"Cows based on your query: {cows}"
                
                # Prepare images
                cow_images = {}
                for cow_id in cows:
                    image_path = os.path.join(image_folder, f"{cow_id}.jpg")
                    if os.path.exists(image_path):
                        cow_images[cow_id] = image_path
                
                return {"text": response_text, "images": cow_images}
            else:
                return {"text": "No matching cows found for your query.", "images": {}}
        except Exception as e:
            return {"text": f"Error executing generated code: {e}", "images": {}}
                
    except Exception as e:
        return {"text": f"Error connecting to OpenAI: {e}", "images": {}}

def process_info_query(query):
    """Handle informational queries about cattle or the system."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": info_system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return {"text": completion.choices[0].message.content, "images": {}}
    except Exception as e:
        return {"text": f"I'm sorry, I couldn't process your informational query: {e}", "images": {}}

def process_conversational_query(query):
    """Handle general conversational queries."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": conversation_system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return {"text": completion.choices[0].message.content, "images": {}}
    except Exception as e:
        return {"text": f"I'm sorry, I couldn't process your conversational query: {e}", "images": {}}

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"sender": "user", "text": user_input})
    st.chat_message("user").write(user_input)
    
    # Classify the query
    query_type = classify_query(user_input)
    
    # Process based on query type
    if query_type == "data_query":
        response = process_data_query(user_input)
    elif query_type == "info_query":
        response = process_info_query(user_input)
    else:  # conversation
        response = process_conversational_query(user_input)
    
    # Add bot response to chat history
    st.session_state.messages.append({
        "sender": "bot", 
        "text": response["text"],
        "images": response.get("images", {})
    })
    
    # Display the text response
    bot_message = st.chat_message("bot")
    bot_message.write(response["text"])
    
    # Display images if any
    if response.get("images", {}):
        cols = st.columns(min(4, len(response["images"])))
        for i, (cow_id, img_path) in enumerate(response["images"].items()):
            col_idx = i % 4
            with cols[col_idx]:
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    st.image(image, caption=f"{cow_id}", use_container_width=True)
                else:
                    st.write(f"{cow_id}: Image not found")