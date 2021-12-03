from os import write
import streamlit as st
 

import numpy as np
import pickle
import numpy as np
import pandas as pd
## import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pdfplumber


header = st.beta_container()
body = st.beta_container()
classify_container = st.beta_container()


# Classification   code 
def classify(a):
    filename = 'sgdmodel_.pkl'
    model_reloaded = pickle.load(open(filename, 'rb'))
    
    te =[]
    te.append(a)
    ab = model_reloaded.predict_proba(te)
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    result = ab.tolist()
    test_res = result[0]
    li_goals = ["No Poverty","Zero Hunger","Good Healthand Well Being","Quality Education"
            ,"Gender Equality","Clean Water and Sanitation","Affordable  and Clean Energy"
            ,"Decent Work and Economic Growth","Industry,Innovation and Infrastructure"
            ,"Reduced Inequalites","Sustainable Cities and Communities",
            "Responsible Consumption and Production","Climate Action","Life Below Water","Life On Land"]
    t =zip(li_goals,test_res)
    df_predic = pd.DataFrame(t,columns=["SDG Category","Probability"])
    df_predic.index = df_predic.index + 1
    fi= df_predic.sort_values("Score", ascending = [False])
    return((fi))
    
def pdf_read(file_data):
     dt = []
     with pdfplumber.load(file_data) as pdf:
     	pages = pdf.pages
     	for page in pages:
     		dt.append(page.extract_text())
     	st = ' '.join(dt)
     return st
       


with header:
    titl, imga = st.beta_columns(2)
    titl.title('SDG Classifier')
    imga.image('1sdg_logo.svg.gif')
    
   

with body:
    rawtext = st.text_area('Enter Text Here')
    

    sample_col, upload_col,pdf_up = st.beta_columns(3)
    
    sample_col.subheader('  [OR]  ')
    sample = sample_col.selectbox('Or select a sample file',
                                  ('AsianPaints19_CSR.txt', 'BharatPetroleum19_CSR.txt','BhartiAirtel17_CSR.txt','HdfcBank16_CSR.txt','None'), index=4)
    if sample != 'None':
        file = open(sample, "r", encoding='utf-8')
        #st.write(file)
        rawtext = file.read()

    upload_col.subheader('  [OR]  ')
    uploaded_file = upload_col.file_uploader(
        'Choose your .txt file', type="txt")
    
    if uploaded_file is not None:
        rawtext = str(uploaded_file.read(), 'utf-8')
    
    pdf_up.subheader('[OR]')
    uploaded_file = pdf_up.file_uploader(
        'Choose your .pdf file', type="pdf")
        
    if uploaded_file is not None:
        rawtext = pdf_read(uploaded_file)    
    
    if st.button('Sdg Classification Results'):
        with classify_container:
            if rawtext == "":
                st.header('Classification :)')
                st.write('Please enter text or upload a file to see the Classification')
            else:
                result = classify(rawtext)
                st.header('Sdg Classification :)')
                #res, plot = st.beta_columns(2)
                st.dataframe(result)
                df = pd.DataFrame(result, columns = ["Probability"])
                st.bar_chart(df)

                st.header('Report:')
                expand = st.beta_expander("Expand to see orignal Report")
                with expand:
                	st.write(rawtext)
    expand_goal = st.beta_expander("United Nation Organisation Sustainable Development Goals [Expand to see]")
    with expand_goal:
     st.image('all_goals.jpg')
