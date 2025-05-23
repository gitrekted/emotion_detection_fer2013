# Emotion Detection using CNN (High School Project)

This was a fun little project I did for a tech competition back in highschool. It uses a Convolutional Neural Network (CNN) to detect emotions from facial expressions in real time using a webcam.

The model is trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and recognizes 7 emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

> ⚠️ Note: This code was written during high school, so it isn't modular and lacks exception handling. But it works, or atleast it used to

---

## 👨‍💻 What It Does

* Trains a CNN model on the FER-2013 dataset using TensorFlow 2.x and Keras.
* Uses OpenCV to open the webcam and detect faces.
* Predicts emotions and displays them on the live webcam feed.

---

## 🗂️ Folder Structure

```
.
├── dataset/
│   └── fer2013.csv           # FER-2013 CSV (you'll need to download this yourself)
├── model/
│   └── emotion_model.h5      # Saved CNN model (generate it using the train_model.py)
├── train_model.py            # Script to train the CNN
├── detect_emotion.py         # Script to detect emotion from webcam
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 🧠 How to Train the Model

1. Download the FER-2013 CSV file from [here](https://www.kaggle.com/datasets/msambare/fer2013) and put it in the `dataset/` folder.
2. Run the training script:

```bash
python train_model.py
```

3. The trained model will be saved to `model/emotion_model.h5`.

---

## 📷 How to Run the Emotion Detector

Make sure you have a webcam.

```bash
python detect_emotion.py
```

Press `q` to quit the webcam window.

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🤓 Notes from My 2021 Self

* I had to use an older version of TensorFlow 2.x because newer ones lagged on my Pentium N5000.
* Tried adding emotion logging but removed it because file writes were literally crashing my potato

---

## 📬 Contact

Feel free to reach out if you have questions, or just want to roast my 2021 code!
