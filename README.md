
# Cyber-Attack Detection System

## Overview

This project implements a machine learning based network intrusion
detection system.

The system analyzes network traffic data and classifies connections
as either normal traffic or potential cyber attacks.

The project uses the UNSW-NB15 dataset and combines an autoencoder
for feature extraction with Random Forest, Support Vector Machine,
and a stacking ensemble classifier.

---

## Problem Statement

Modern networks generate a large amount of traffic every second.
Manually examining this traffic is difficult and time-consuming.

The objective of this project is to use machine learning to identify
patterns in network traffic and automatically classify the traffic
as normal or malicious.

---

## Objectives

- Detect potentially malicious network traffic.
- Preprocess real-world network traffic data.
- Extract useful features using an autoencoder.
- Train multiple machine learning classifiers.
- Combine classifiers using a stacking ensemble.
- Evaluate the model using standard classification metrics.

---

## Dataset

The project uses the UNSW-NB15 network intrusion detection dataset.

The dataset contains network traffic records with different
network and connection characteristics.

The dataset files are not included in this repository.

Required files:

```text
data/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
