# 🧑 Face Detection using OpenCV DNN

A professional Face Detection application built with **Python**, **OpenCV**, and **Streamlit**.  
It uses a pre-trained Deep Learning model (ResNet-10 SSD from OpenCV) to detect faces in images and videos with high accuracy.

The app supports:
- Uploading images (JPG, PNG)
- Uploading videos (MP4, AVI, MOV)
- Adjustable confidence threshold
- Real-time bounding boxes with confidence scores

Deployed live on Streamlit Cloud (free): [Add your deployed link here after deployment]

## 🚀 Features

- Fast and accurate face detection using OpenCV's pre-trained DNN model
- Clean and user-friendly Streamlit interface
- Support for both static images and recorded videos
- Confidence threshold slider for fine-tuning results
- Automatic model download on first run
- Fully open-source and easy to deploy

## 📁 Project Structure

```
face-detection-opencv-streamlit/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── model/                  # Auto-created: contains downloaded model files
├── colab_notebook.ipynb    # Google Colab notebook for development & testing
├── screenshots/            # Folder for demo images (optional)
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🛠 Tech Stack

- **Python 3.8+**
- **OpenCV** (DNN module with Caffe model: res10_300x300_ssd_iter_140000)
- **Streamlit** for the web interface
- **NumPy** & **Pillow** for image processing

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-detection-opencv-streamlit.git
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

4. Open your browser at 

The model will be downloaded automatically on first run (~10.7 MB).

## 📓 Google Colab Notebook

A complete development notebook is included:  
(HexSoftware_Face_Detection.ipynb)

It contains:
- Model loading and testing
- Face detection on sample images from datasets
- Video processing examples
- Full experimentation environment

Perfect for learning or extending the project.

## 🌍 Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repo
5. Set:
   - Branch: `main`
   - File: `app.py`
6. Click "Deploy"

Your app will be live in minutes!

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this project for personal or commercial purposes.

## 👨‍💻 Author

Eslam Alsaeed
- GitHub: https://github.com/eslamalsaeed72-droid
- LinkedIn: https://www.linkedin.com/in/eslam-alsaeed-1a23921aa 

## 🙏 Acknowledgments

- OpenCV team for the excellent DNN face detector
- Streamlit for the amazing framework
- All open-source contributors

---

⭐ If you like this project, give it a star on GitHub!  
Feel free to open issues or contribute improvements.

Happy detecting! 🧑
```
