import tempfile
import streamlit as st
import tools.infer.utility as utility

from tools.infer.predict_system import inference

def main():
    st.title('Paddle OCR')
    st.write('Optical character recognition for simplfied Chinese and English.')
    st.header('Upload your image')
    image = st.file_uploader('', type=['png', 'jpg', 'jpeg'])
    
    if image is not None:
        img = tempfile.NamedTemporaryFile(delete=True)
        img.write(image.read())

        st.image(image)
        args = utility.parse_args()
        args.vis_font_path = 'tools/msjh.ttf'
        args.image_dir = img.name
        pred_image, texts = inference(args)

        st.header('Inference result')
        st.image(pred_image)
        st.subheader('Texts')
        for line in texts:
            st.code(line, language=None)
    
    st.header('Reference')
    st.markdown('This app is developed by [Justin Bear](https://github.com/JustinBear99/PPOCR-app) using [PP-OCRv2](https://github.com/PaddlePaddle/PaddleOCR) model.')

if __name__ == "__main__":
    main()