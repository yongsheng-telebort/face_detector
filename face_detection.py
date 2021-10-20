import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv

st.title('Face Detector')

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
#         width:350px
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
#         width:350px
#         margin-left: -350px
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

st.sidebar.title('Face Detector Sidebar')
st.sidebar.subheader('Pages')


@st.cache()
def detect_image(image):
    ###
    pass


app_mode = st.sidebar.selectbox(
    'Choose', ['Home', 'About me', 'Try the app'])

if app_mode == 'Home':
    st.markdown(
        'This face detction app is using Haar Cascades Classifier under OpenCV')

    st.video('https://www.youtube.com/watch?v=hPCTwxF0qf4')


elif app_mode == 'About me':
    st.markdown(
        '''
        ### Name: Yong Sheng \n
        ### School: Telebort \n
        ### Age: 15 \n
        ''')

elif app_mode == 'Try the app':
    # st.sidebar.markdown('''
    #     Choose A Number
    # ''')
    st.markdown('**Detected Faces**', unsafe_allow_html=True)
    # kpi1_text = st.markdown('0')

    # max_faces = st.sidebar.number_input(
    #     'Maximum Number of Face', value=2, min_value=1)
    # confidence = st.sidebar.slider(
    #     'Confidence', value=0.5, min_value=0.0, max_value=1.0)
    image_file = st.sidebar.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        st.sidebar.text('Original Image')
        st.sidebar.image(image)

        # Detection
        # Phase 2: Load trained data
        trained_face_data = cv.CascadeClassifier(
            "haarcascade_frontalface_default.xml")

        # Phase 3: Image Processing for Detection
        # Read image
        # img = cv.imread("outing.jpg")

        # resize image
        # scale = 0.2
        # width = int(img.shape[1]*scale)
        # height = int(img.shape[0]*scale)
        # dim = (width, height)
        # img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        # Change image to gray
        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Phase 4: Face Detection
        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(gray_img)

        # To draw rectangle for each face
        for face in face_coordinates:
            (x, y, w, h) = face
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)

        # Phase 5: Show image
        st.subheader('Output Image')
        st.image(image, use_column_width=True)
    else:
        pass
