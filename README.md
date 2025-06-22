# Contents

1.  [Repository Name](#repository-name)
2.  [Title of the Project](#title-of-the-project)
3.  [Short Description and Objectives of the Project](#short-description-and-objectives-of-the-project)
4.  [Details about the Datasets](#details-about-the-datasets)
5.  [Goal of this Project](#goal-of-this-project)
6.  [Project Requirements](#project-requirements)
7.  [Usage Instructions in Local System and Google Colab](#usage-instructions-in-local-system-and-google-colab)
8.  [Detailed Conclusion with points](#detailed-conclusion-with-points)
9.  [Key Takeaways](#key-takeaways)
10. [Recommendation for this project](#recommendation-for-this-project)
11. [Future Improvements](#future-improvements)
12. [Deployment of the trained best model](#deployment-of-the-trained-best-model)
13. [Final Thoughts](#final-thoughts)
14. [Authors](#authors)
15. [References](#references)
----------------------------------------------

# Repository Name
unspoken-signals-voice-based-ai-emotion-detection

----------------------------------------------

# Title of the Project
Unspoken Signals: Voice-Based AI to Decode Emotions

----------------------------------------------

# Short Description and Objectives of the Project
- Thoughts and feelings are shared between humans not only through verbal language but also through the nuanced variations of human vocalization - variations in emphasis (tone), loudness (energy), pitch, and timing (rhythm) - signals that machines are often not adept at interpreting. While machines have become remarkably more intelligent, our capabilities for perceiving and interpreting the emotional aspects of spoken human language remain far behind technology. This is highly relevant when developing empathetic AI systems that consider the emotional aspect of human affect, such as monitoring mental health, dealing with customers, or human-computer interaction. 
  
- ***Unspoken Signals*** will tackle this issue using an AI driven application that can decode vocal signals and provide an interpretation of just about any emotional type. The project will utilize RAVDESS Emotional Speech Audio dataset and continue to explore how dimensions of emotional expression correlate with various acoustic features - with the hope to build some models that will be able to discover not just the emotional expression, but also the acoustic representations of emotions through voiced and vocalized language. It is not only about establishing classification categories for emotions like happiness, sadness, anger, fear, etc, but making the implicit meaning of how acoustic features can translate to emotional meaning, therefore getting machines one step further to developing some form of intuitive emotional intelligence.

----------------------------------------------

# Details about the Datasets

- **Name of the Dataset:** RAVDESS Emotional Speech Audio
- **Description of the dataset:** Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS. 

- Full dataset of speech and song, audio and video (24.8 GB) available from Zenodo. Construction and perceptual validation of the RAVDESS is described in our Open Access paper in PLoS ONE.
	- Zenodo Reference: https://zenodo.org/records/1188976
	- PLoS ONE Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391 

- **Dataset Files:** This portion of the RAVDESS contains 1012 files: 44 trials per actor x 23 actors = 1012. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Song emotions includes calm, happy, sad, angry, and fearful expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

- **File naming conventions:** Each of the 1012 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-02-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

- **Filename identifiers:**

  - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  - Vocal channel (01 = speech, 02 = song).
  - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  - Repetition (01 = 1st repetition, 02 = 2nd repetition).
  - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


- **Filename example:** 03-02-06-01-02-01-12.wav

  - Audio-only (03)
  - Song (02)
  - Fearful (06)
  - Normal intensity (01)
  - Statement "dogs" (02)
  - 1st Repetition (01)
  - 12th Actor (12)
  - Female, as the actor ID number is even.

----------------------------------------------

# Goal of this Project
- ***Unspoken Signals*** is a project to teach machines to learn the emotional richness, expressed through speech, in the way we interpret, without thought of joy, sadness, anger, and a kingdom of others in another person's tone. While many speech/emotion systems simply classify furthermore understand emotion in speech, we aim to analyze the voice to find the elements of the voice, such as pitch, pace, energy, and rhythm, to understand the emotional pattern. We want to look beyond the classification of emotion and towards why the specific characteristics of the voice create specific emotional responses. 

- ***Unspoken Signals*** will create a flow between sound and sentiment, in the way that we flow as humans to interpret facial signals to vocal signals everyday. This initiative aims to bring machines to an intuition of the richness or depths of emotion's impact to comprehend the intangibility of emotion from sound, vocals alone.

----------------------------------------------

# Project Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- joblib
- itertools
- json5
- librosa
- pyloudnorm
- xgboost
- lightgbm
- torch
- torchaudio
- tensorflow
- ipython
- notebook
- warnings

----------------------------------------------

# Usage Instructions in Local System and Google Colab
- Clone using HTTPS
```commandline
[https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs.git](https://github.com/kraviteja95usd/unspoken-signals-voice-based-ai-emotion-detection.git)
```
OR - 

- Clone using SSH
```commandline
git@github.com:kraviteja95usd/unspoken-signals-voice-based-ai-emotion-detection.git
```

OR -

- Clone using GitHub CLI
```commandline
gh repo clone kraviteja95usd/unspoken-signals-voice-based-ai-emotion-detection
```
 
- Switch inside the Project Directory
```commandline
cd unspoken-signals-voice-based-ai-emotion-detection
```

- Install Requirements in your local
```commandline
pip3 install -r requirements.txt
```

- Follow these guidelines to execute in Google Colab
  - Download the dataset zip file from [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download).
  - Rename the downloaded dataset and upload it to your Google Drive.
  - Open the [colab notebook](https://github.com/kraviteja95usd/unspoken-signals-voice-based-ai-emotion-detection/blob/main/Ravi%20Teja%20Kothuru_Final%20Project%20AAI-510-Section%20IN2-Team%2014.ipynb) into your Colab account.
  - Change the **Extract Audio Dataset** section with the path to the dataset zip file you saved.
  - Run the entire colab notebook. You will be able to view the results.


----------------------------------------------

# Detailed Conclusion with points

- Best Classical ML Model: `Extra Trees`
  - Achieved **highest accuracy (79.67%)**, **precision (80.05%)**, and **F1-score (79.65%)**
  - Lowest loss (0.20) among all models.
  - Highly interpretable, stable, and fast to train.

- Best Deep Learning Model: `BN_Activated_DeepNN`
  - Second-best in accuracy (78.67%) and F1-score (74.97%)
  - Performed robustly due to batch normalization, dropout, and deeper architecture.
  - Particularly suitable for more complex data or unstructured input (e.g., images, sequences).

- Final Winner: `Extra Trees`
  - Outperforms deep learning models on **all reported metrics**
  - Recommended for deployment due to ease of use, interpretability, and minimal tuning needs.

----------------------------------------------

# Key Takeaways

| Aspect                  | Classical ML (Best: Extra Trees)      | Deep Learning (Best: BN_Activated_DeepNN)       |
|-------------------------|----------------------------------------|--------------------------------------------------|
| Accuracy (%)            | **79.67%**                           | 78.67%                                           |
| F1 Score (%)            | **79.65%**                           | 74.97%                                           |
| Precision (%)           | **80.05%**                           | *NA*                                   |
| Recall (%)              | **79.67%**                           | *NA*                                   |
| Loss                    | **0.20**                             | *NA*                                   |
| Interpretability        | High                                 | Low                                            |
| Training Time           | Low                                  | Higher                                         |
| Tuning Complexity       | Low                                  | Medium to High                                 |
| Generalization          | Strong                               | Strong (with regularization)                  |

 - Acoustic attributes (MFCCs, energy, pitch, ZCR) are strong representations of emotional patterns in speech.
 - The ensemble model ExtraTrees did a good job of emotion classification on structured audio features.
 - Deep learning models are weakly promising with generalization and regularization, but we need more time to train and fine-tune and need more data. 
 - Using StandardScaler preprocessing, helped build a more stable and higher-performing model overall. 

----------------------------------------------

# Recommendation for this project

- Use **Extra Trees** as the default model for structured/tabular datasets.
- Use **BN_Activated_DeepNN** if:
  - The dataset grows significantly in size.
  - You're dealing with more complex patterns or plan to use transfer learning.
  - Additional precision/recall tuning isn’t a bottleneck.

----------------------------------------------

# Deployment of the trained best model

***Justification for Not Fully Deploying the Model:***

While full deployement of the emotion detection (via a voice assistant or a chatbot) would have engaged us far more hands-on in practical learnings, a broader set of constraints led us to focus on deployment planning as a conceptual exercise:

  - **Time Constraints:** I spent a substantial amount of project time in woman hours in investing in building, validating, and analyzing both classical ML and DL models. The time required for foraml deployment, with any expectation of genuine real-time scenarios, would also include investing additional time in containerization, API developent, UI integration, and cloud hosting. 
  
  - **Resource Constraints:** Real-time model deployment with live audio input involves consistent and high-performing compute environment - especially with DL inference scenarios - while local resources could not adequately provide real-time processing with a low-latency experience, leaning into cloud deployment would bring adds to setup overhead and potentially costs.
  
  - **Focus on Model Evaluation:** Given our objective was to choose and evaluate the best model amongst a multiplicity of approaches (ML vs. DL), I prioritized robustness to facilitate metric comparisons and interpretability. I was able to conduct this depth of analysis to fulfill the learning objectives of the project, and to identify clear trade-offs which will be intuitive in AI development in the real world.

- **Hypothetical Deployment Plan (Fulfilling Assignment Requirement):**

Although not a deployment, a full deployment scenario is planned and described below as required by the project deployment description: 

| **Component**       | **Plan**                                                                 |
|---------------------|--------------------------------------------------------------------------|
| **Use Case**        | Real-time emotion detection using a chatbot or voice assistant application.               |
| **Deployment Type** | **Real-Time Inference** - Immediate return of emotion inference on audio input.       |
| **Backend**         | Python API with **FastAPI** or **Flask** to serve the trained model.     |
| **Frontend**        | A web interface or mobile application with live audio input through the use of WebRTC or HTML5.   |
| **Hosting**         | The model hosted on **AWS EC2** or with **Azure App Service docker** support. 
| **Latency Target**  | Less than 300 ms inference time to allow for smooth interaction with the user.                       |
| **Cost Consideration** | Initial deployment using a low-tier cloud instance (i.e., AWS or Azure) and autoscaling later if needed.  |                |
| **Security**        | Token-based API authentication, HTTPS frontend, and CORS policy.       |

- Deployment related Summary

> Although full deployment was not completed due to logistical complications, a cloudy and technically feasible deployment plan was identified. This addressed the deployment component of the project, proving readiness for the role of real-world integration of the model into production systems.

----------------------------------------------

# Future Improvements

- Use more datasets, like `CREMA-D` and `TESS`, to improve generalization.
- Make it multimodal and try to extract and classify emotions from text, image and video inputs as well.
- Test audio-based Transformers or CNN-RNN hybrid models for end-to-end learning with raw waveforms.  
- Use `GridSearchCV` or `Optuna` for machine learning models.  
- Use `Keras Tuner` or `Ray Tune` for deep learning models.  
- Combine top classical and deep learning models through stacking or soft voting to enhance robustness.  
- Apply stratified k-fold validation to ensure the model works well across different folds.  
- Use SHAP or permutation importance to refine feature inputs.  
- Measure inference time, memory use, and latency.  
- Wrap/Deploy the model in an API using FastAPI or Flask for testing and inferencing.

----------------------------------------------

# Final Thoughts

While the potential of deep learning exists, particularly as new methods of architecture are developed, **traditional ML models are still very competitive** against structured datasets. As a result of lesser resources and less complexity in implementing it, it can frequently achieve **near top-tier results**, as was demonstrated here.  

> Use the tool appropriate to the problem — not everything requires neural networks.

----------------------------------------------

# Authors

| Author            | Contact Details       |
|-------------------|-----------------------|
| Ravi Teja Kothuru | rkothuru@sandiego.edu |

----------------------------------------------

# References
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *Zenodo (CERN European Organization for Nuclear Research)*. https://doi.org/10.5281/zenodo.1188975
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE, 13*(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

‌


