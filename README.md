# Human Activity Recognition with a Hidden Markov Model

An individual project that infers human activity (standing, walking, jumping, still) from smartphone accelerometer and gyroscope data, using a Hidden Markov Model built from scratch with NumPy.

## Overview

- **Data:** my own recordings made with the Sensor Logger app (accelerometer and gyroscope, x/y/z).
- **Pre-processing:** the phone delivered about 52.7 Hz (100 Hz was set), so both sensors were aligned and resampled to 50 Hz, then cut into 2 s windows with 50% overlap.
- **Features (8 per window):** accelerometer magnitude std, accelerometer SMA, gyroscope magnitude std, mean axis correlation (time domain), and accelerometer dominant frequency, spectral energy, spectral entropy and gyroscope dominant frequency (FFT). Log10 on the wide-range features, then Z-score fitted on training data only.
- **Model:** Gaussian HMM with 4 hidden states (the activities) and diagonal-covariance Gaussian emissions.
- **Algorithms:** Baum-Welch (EM, stops when the log-likelihood change is below 1e-3) for training and Viterbi for decoding.

## Results on unseen data

Two new recordings, each with all four activities back to back, recorded in a different place and session, with the phone held lower than in training. 73 windows in total.

- Overall accuracy: **0.808** (0.902 when the 12 windows that straddle an activity switch are excluded)

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Standing | 17 | 0.706 | 0.929 | 0.877 |
| Walking | 20 | 0.900 | 0.925 | 0.918 |
| Jumping | 18 | 1.000 | 0.891 | 0.918 |
| Still | 18 | 0.611 | 1.000 | 0.904 |

Jumping and walking were easiest to recognize. Standing and still were hardest, and most errors happened within a few seconds of an activity switch.

## Repository contents

```
hidden_markov_models_notebook.ipynb   # full notebook with explanations and analysis
figures/                          # plots used in the report
results /                        # test results
README.md
```

[Add the report file name here, for example `report.pdf`, if you include it.]

## Figures

- `sample_signals.png`: accelerometer and gyroscope sample per activity
- `transition_and_ll.png`: transition matrix heatmap and Baum-Welch log-likelihood
- `emissions.png`: emission means per state
- `test_signals.png`: unseen test signals with ground-truth switch times
- `confusion_matrix.png`: confusion matrix on unseen data
- `decoded_sequences.png`: true vs predicted activity sequences

## How to run

The notebook was written for Google Colab with data in Google Drive.

1. Upload the Sensor Logger exports to `MyDrive/hmm_project/`:
   - `raw_zip_files/` holds the training recordings, named `Standing_01...zip`, `Walking_01...zip`, `Jumping_01...zip`, `Still_01...zip` (the label is read from the start of the file name).
   - `test_files/` holds the two unseen recordings (`Test_file_01...zip`, `Test_file_02...zip`).
2. Open the notebook in Colab and run the cells from top to bottom.
3. The notebook creates a `clean/` folder and saves features, model parameters and figures in `hmm_project/`.

**Requirements:** Python 3, numpy, pandas, scipy, matplotlib (all preinstalled in Colab).

The ground-truth timing of the two test files is set in the `truth` cell of the notebook.

## Demo videos

Video of me recording each activity (entire body visible):

- Standing: [link]
- Walking: [link]
- Jumping: [link]
- Still: [link]

## Limitations

- One participant, one phone and one training session; the test set has only 73 windows.
- Each training clip contains a single activity, so training sessions were built by chaining one clip per activity in random order. The off-diagonal transition probabilities therefore come from that random ordering and do not describe real behavior.
- Baum-Welch was initialized from the labels, so training accuracy (1.000) is not a measure of generalization.
- Only one sampling rate (50 Hz) and one window size (2 s) were tested.

## Author

Sheilla Keza RUVUGABIGWI, African Leadership University
