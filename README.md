
# 🧑 Face Detection using OpenCV DNN

A professional **Face Detection** application built with **Python**, **OpenCV**, and **Streamlit**.  
It utilizes OpenCV's pre-trained Deep Learning model (ResNet-10 SSD) to accurately detect faces in both images and videos.

**Live Demo:**  https://ezbewt64biwjtn6u2ykj63.streamlit.app/

## 🚀 Features

- Fast and accurate face detection using OpenCV's pre-trained DNN module
- Support for **image** (JPG, PNG, JPEG) and **video** (MP4, AVI, MOV) uploads
- Frame-by-frame video processing with progress tracking
- Interactive **confidence threshold slider** for fine-tuning results
- Automatic download of the pre-trained model on first run
- Clean, responsive, and user-friendly Streamlit web interface
- Fully open-source and deployable on Streamlit Cloud (free)

## 📁 Project Structure

```
face-detection-opencv-streamlit/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies (uses opencv-python-headless)
├── colab_notebook.ipynb          # Full Google Colab development & testing notebook
├── Demo/                         # Demo video
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

## 🛠 Technologies & Tags

- **Python**
- **OpenCV** (DNN module)
- **Streamlit**
- **NumPy**
- **Pillow**
- **Deep Learning** (pre-trained Caffe model: res10_300x300_ssd_iter_140000)
- **Computer Vision**
- **Face Detection**
- **Web Application**
- **Google Colab**
- **GitHub**
- **Streamlit Cloud Deployment**
- **AI / Machine Learning**
- **Portfolio Project**

#Python #OpenCV #ComputerVision #DeepLearning #Streamlit #FaceDetection #AI #MachineLearning #DataScience #PythonProgramming #WebApp #Deployment #GitHub #GoogleColab #Portfolio #Tech

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/eslamalsaeed72-droid/HexSoftware_Face_Detection.git
   cd face-detection-opencv-streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

The model (~10.7 MB) will be downloaded automatically on first run.

## 📓 Google Colab Notebook

Full development and experimentation notebook:  
[**colab_notebook.ipynb**](HexSoftware_Face_Detection.ipynb) :
https://colab.research.google.com/drive/1KJRkD_gs1LFmLL4XmhkXFExfVNR46Z8L?usp=drive_link

Includes:
- Model loading and validation
- Testing on large datasets
- Video processing examples
- Step-by-step comments

## 🌍 Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. New app → Connect your repo
4. Set main file: `app.py`
5. Deploy!

> Tip: Use `opencv-python-headless` in requirements.txt for smooth deployment.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Eslam Alsaeed 
- GitHub: https://github.com/eslamalsaeed72-droid 
- LinkedIn: https://www.linkedin.com/in/eslam-alsaeed-1a23921aa 

## 🙏 Acknowledgments

- OpenCV team for the powerful DNN face detector
- Streamlit for the incredible web framework
- HexSoftware AI Track for the hands-on learning experience

---

⭐ Star this repo if you found it useful!  
Feel free to fork, contribute, or open issues.

Happy detecting! 🧑
```
