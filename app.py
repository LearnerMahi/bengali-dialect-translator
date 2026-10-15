import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your model
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("./final_bengali_model")
    tokenizer = AutoTokenizer.from_pretrained("./final_bengali_model")
    return model, tokenizer

model, tokenizer = load_model()

LANGUAGES = ["Standard_Bangla", "Chittagong", "Sylhet", "Barisal", "Noakhali", "Mymensingh", "Rangpur", "Kishoreganj"]

def translate(text, source_lang, target_lang):
    if not text.strip():
        return "Please enter some text."
    input_text = f"translate {source_lang} to {target_lang}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("Bengali Dialect Translator")
st.write("Translate between Standard Bangla and regional dialects using a fine-tuned mT5 model.")

source_lang = st.selectbox("Source Language", LANGUAGES, index=1)
target_lang = st.selectbox("Target Language", LANGUAGES, index=0)
text = st.text_area("Input Sentence", height=100)

if st.button("Translate"):
    result = translate(text, source_lang, target_lang)
    st.success(result)