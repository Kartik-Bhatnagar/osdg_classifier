
from os import write
import streamlit as st
 

import numpy as np
import pickle
import numpy as np
import pandas as pd
## import plotly.figure_factory as ff
import matplotlib.pyplot as plt


header = st.beta_container()
body = st.beta_container()
classify_container = st.beta_container()



######################## Summarization code  ########################################


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
    df_predic = pd.DataFrame(t,columns=["OSDG","Score"])
    df_predic.index = df_predic.index + 1
    fi= df_predic.sort_values("Score", ascending = [False])
    return((fi))
    


with header:
    titl, imga = st.beta_columns(2)
    titl.title('SDG Classifier')
    imga.image('1sdg_logo.svg.gif')
    
   

with body:
    st.header('Sdg classification')
    rawtext = st.text_area('Enter Text Here')
    

    sample_col, upload_col = st.beta_columns(2)
    
    sample_col.subheader('[Or] select a sample file from below')
    sample = sample_col.selectbox('Or select a sample file',
                                  ('AsianPaints19_CSR.txt', 'BharatPetroleum19_CSR.txt','BhartiAirtel17_CSR.txt','HdfcBank16_CSR.txt','None'), index=4)
    if sample != 'None':
        file = open(sample, "r", encoding='utf-8')
        #st.write(file)
        rawtext = file.read()

    upload_col.subheader('[Or] upload text file here')
    uploaded_file = upload_col.file_uploader(
        'Choose your .txt file', type="txt")
    if uploaded_file is not None:
        rawtext = str(uploaded_file.read(), 'utf-8')

    #no_of_lines = st.slider("Select number of lines in summary", 1, 5, 3)
    if st.button('Get Results'):
        with classify_container:
            if rawtext == "":
                st.header('Classification :)')
                st.write('Please enter text or upload a file to see the Classification')
            else:
                result = classify(rawtext)
                st.header('Sdg Classification :)')
                res, plot = st.beta_columns(2)
                res.dataframe(result)
                df = pd.DataFrame(result, columns = ["Score"])
                #result.hist()
                #plt.show()
                #st.pyplot()
                plot.bar_chart(df)

                # Abstractive summary
                #st.header('Abstractive method')
                #abstract = abstractive(rawtext)
                # st.write(abstract)
                st.header('Report:')
                expand = st.beta_expander("Expand to see orignal Report")
                with expand:
                	st.write(rawtext)
    st.image('all_goals.jpg')
