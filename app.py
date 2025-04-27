import streamlit as st
from openai import OpenAI
import pandas as pd
import re
import os
from PIL import Image

# Cow_id images folder
image_folder = r"D:\ECE\4th sem\WisEST\MooBot\code\cow_images_optimized" 

# Load the master schmea

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI clientW

# App title
st.set_page_config(page_title="MooBot Chatbot")

st.title("🐄 MooBot: Dairy Farm Chatbot")
st.write("Ask me anything about your cattle!")

# Load the master schema CSV (ensure your CSV file path is correct)
try:
    # st.write("Loading master schema CSV...")
    master_schema = pd.read_csv("combined_intersection_df.csv")  # Replace with your actual path
    # st.write("Master schema loaded successfully!")
except Exception as e:
    st.write(f"Error loading master schema: {e}")
combined_intersection_df = master_schema # dataframe name used in system prompt

# Chat message storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["sender"]).write(msg["text"])

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"sender": "user", "text": user_input})
    st.chat_message("user").write(user_input)

    # Define system prompt
    system_prompt ="""You are an expert in Python and Pandas. Generate valid and efficient Pandas code to extract information from a dataframe called `combined_intersection_df` using the user's query.

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
    """
    # Initialize bot_response with a default value
    bot_response = "I couldn't process your query."
    
    # Send user query to OpenAI for processing
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        # Get the generated code from OpenAI response
        generated_code = completion.choices[0].message.content

        # Extract only the Python code part by removing the explanation text and markdown
        python_code = re.search(r'```python\n(.*?)```', generated_code, re.DOTALL)
        if python_code:
            python_code = python_code.group(1)  # Extract the code inside the code block
        else:
            st.write("Error: No Python code found in the response.")
            bot_response = "Error: No Python code found in the response."
            python_code = ""

        # Inspect and debug generated code
        if python_code:
            try:
                # Safely execute the cleaned Python code within the Streamlit app environment
                exec(python_code)
                
                # If cows list is generated, prepare and display results
                if 'cows' in locals():
                    # Create a text response
                    bot_response = f"Cows based on your query: {cows}"
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"sender": "bot", "text": bot_response})
                    st.chat_message("bot").write(bot_response)
                    
                    # Display cow images
                    st.write("### Cow Images:")
                    cols = st.columns(min(4, len(cows)))  # Create up to 4 columns
                    
                    for i, cow_id in enumerate(cows):
                        col_idx = i % 4  # Determine which column to place the image
                        with cols[col_idx]:
                            try:
                                # Construct image path
                                image_path = os.path.join(image_folder, f"{cow_id}.jpg")
                                
                                if os.path.exists(image_path):
                                    image = Image.open(image_path)
                                    st.image(image, caption=f"{cow_id}", use_container_width=True)
                                else:
                                    st.write(f"{cow_id}: Image not found")
                            except Exception as e:
                                st.write(f"Error loading image for {cow_id}: {e}")
                    
                    # # Skip adding the response again at the end of the function
                    # return
                else:
                    bot_response = "No matching cows found for your query."
            except Exception as e:
                st.write(f"Error executing generated code: {e}")
                bot_response = f"Error executing code: {e}"
                
    except Exception as e:
        st.write(f"Error connecting to OpenAI: {e}")
        bot_response = f"Error: {e}"

    # Append bot response to chat history (only for text responses or errors)
    st.session_state.messages.append({"sender": "bot", "text": bot_response})
    st.chat_message("bot").write(bot_response)