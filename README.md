# Phoneme classification and SEG generation

This project implements a neural pipeline for automatic phoneme class classification from short audio recordings (one or two words).

The system predicts, for each time segment of the signal, whether it
corresponds to:
- vowel
- voiced consonant
- voiceless consonant
- sonorant

Based on these predictions, the pipeline automatically generates a `.seg` file with phoneme class labels.

## Project task

Input:
- an audio signal containing one or two spoken words (from CORPRESS)

Output:
- a `.seg` file with time-aligned phoneme class labels (Vowel / Voiced consonant / Unvoiced consonant / Sonorant)

The project also includes tools for training, evaluation, and error analysis at the phoneme level.


## Pipeline overview

1. **Audio preprocessing**
   - Reading raw audio files (`.sbl`)
    (`segtools.py`)
   - Feature extraction (log-mel spectrogram + MFCC)
    (`logmel_mfcc.py`)
   - Frame step: 10 ms

2. **Neural model**
   - CRNN architecture (Conv1D + BiLSTM)
     (`model.py`)
   - Frame-level classification into 4 phoneme classes
     (`train.py`, `dataset.py`)

3. **Postprocessing**
   - Conversion of frame-level predictions into continuous segments
     (`infer_to_seg.py`)
   - Detection of segment boundaries at class changes

4. **SEG file generation**
   - Automatic writing of `.seg` files in the required format
     (`segtools.py`, `infer_to_seg.py`)

  
## Model architecture

- Input: acoustic features per frame
- Conv1D layers for local temporal patterns
- Bidirectional LSTM for temporal context
- Fully connected layer with softmax output


## Repository structure

```text
.
├── train.py              # Model training
├── model.py              # CRNN definition
├── dataset.py            # Dataset and padding
├── make_frames.py        # Frame-level labels from SEG
├── logmel.py             # Log-mel feature extraction
├── logmel_mfcc.py        # Log-mel + MFCC feature extraction
├── infer_to_seg.py       # Inference and SEG generation
├── eval_confusion.py     # Evaluation (confusion matrix)
├── error_tables.py       # Phoneme-level error analysis
├── segtools.py           # SEG read/write utilities
├── requirements.txt      # Python dependencies
└── README.md
```

##Environment setup

The project is designed to be run in a virtual Python environment.
All required dependencies are listed in requirements.txt.

```bash

pip install -r requirements.txt
```

##Training
```bash
python train.py
```

##Evaluation
```bash
python eval_confusion.py
```

##SEG generation
```bash
python infer_to_seg.py
```

##Error analysis
```bash
python error_tables.py
```


