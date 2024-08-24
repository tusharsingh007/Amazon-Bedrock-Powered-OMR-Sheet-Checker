import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError
import base64
from io import BytesIO

# Initialize the Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Specify the model ID
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

def get_claude_response(encoded_image):
    # Create the payload with the base64-encoded image and the text prompt
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "You are an intelligent system designed to evaluate OMR (Optical Mark Recognition) answer sheets. An OMR sheet contains questions, each with multiple choice options labeled A, B, C, and D. The student fills in one of these options for each question by marking the corresponding bubble. You will be provided with an image of such an OMR sheet. Your task is to analyze the image and extract the information for each question, specifically identifying which option (A, B, C, or D) the student has marked. Please return the results in a clear, structured format, listing each question number alongside the marked option. If a question has multiple marked options or is left blank, please note that as well."
                    }
                ]
            }
        ],
        "max_tokens": 10000,
        "anthropic_version": "bedrock-2023-05-31"
    }
    
    # Invoke the model and get the response
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        body=json.dumps(payload)
    )
    
    # Read the response
    output_binary = response["body"].read()
    output_json = json.loads(output_binary)
    output = output_json["content"][0]["text"]
    
    return output

# Streamlit app
st.title("OMR Answer Sheet Evaluator")

# Image uploader
uploaded_image = st.file_uploader("Upload an OMR image", type=["jpeg", "jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded OMR Sheet", use_column_width=True)
    
    # Read the image and encode it as base64
    img_bytes = uploaded_image.read()
    encoded_image = base64.b64encode(img_bytes).decode()
    
    # Process the image using the Claude model
    with st.spinner('Processing the image...'):
        result = get_claude_response(encoded_image)
    
    # Display the result
    st.subheader("Claude's Evaluation Result:")
    st.text(result)
