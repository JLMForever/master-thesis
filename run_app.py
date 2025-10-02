import os
import sys
import shutil

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# copy model
for filename in ["ragflow.py", "reflection_model.h5", "tokenizer.pkl", "label_maps.pkl", "config.pkl"]:
    src = resource_path(filename)
    dst = os.path.join(os.getcwd(), filename)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

# start Streamlit
streamlit_script = os.path.join(os.getcwd(), "ragflow.py")
os.system(f'streamlit run "{streamlit_script}"')
