import streamlit as st
from nb_streamlit import model

st.write('''# Spam detection
	Using naive bayes''')
st.subheader('Enter the message:')
msg= [st.text_input('')]
if st.button('Submit'):
	vect = model()[0].transform(msg).toarray()
	my_prediction = model()[1].predict(vect)
	st.write('It\'s spam' if my_prediction==1 else 'It\'s not spam')
