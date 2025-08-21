## Notes
# Promptemplate, adalah compnent yang penting pada langchain, 
# fungsinya seperti resep yang dapat digunakan berulang kali untuk mendefinisikan perintah kepada LLM


from langchain_core.prompts import PromptTemplate

template_string = "Explain this concept simply and concisely: {concepts}"

# Kalimat yang didalam kurung siku mengindikasi kita akan menggunakan nilai masukan (concept) yang dinamis.
# Untuk mengkonversi perintah dalam bentuk string kedalam FormTemplate kita dapat menggunakan method `#!python .from_template()`

prompt_template = PromptTemplate.from_template(
    template=template_string
)

# Untuk mengetahui bagaimana variable disisipkan pada prompt template kita dapat menggunakan method `#!python invoke()` dengan parameter
# pertama adalah promp dalam bentuk `#!python dict`

inserted_template = prompt_template.invoke(
    input={
        "concepts":"Concept substraction"
    }
)
print(inserted_template)

## Integratoin with LLM
# Pertama kita perlu mendefinisikan LLM sebagaimana yang telah kita lakukan
from langchain_huggingface import HuggingFacePipeline as hfp
from huggingface_hub import login

# Load env
import os
from dotenv import dotenv_values as env


# Only run this once
# my_env = env(os.path.abspath("env"))
# access_token_write = my_env.get("ACCESS_TOKEN")
# login(token=access_token_write)

llm = hfp.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
)


# Untuk mengintegrasikan antara LLM dengan prompt template kita akan menggunakan Langchain expression Language LCEL.
# Karakter pipe | membentuk sebuah rantai atau ikatan yang mana komponen penting lainya dari lang Chain.

llm_chain = prompt_template | llm

# Chain | menghubungkan serangkaian panggilan ke berbagai component menjadi satu urutan, misalkan pada contoh selanjutnya,
# perintah atau pertanyaan yang kita berikan akan dilempar kedalam prompt template lalu dilanjutkan ke LLM

response = llm_chain.invoke(
    {"concepts": "Hallo, good night"}
)

print(response)