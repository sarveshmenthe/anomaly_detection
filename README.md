# anomaly_detection
Frame-level video anomaly detection on the Avenue dataset with an AP-driven, ranking-aware analysis of spatial autoencoder limits.
Anomaly Detection
Sarvesh Menthe
Chemical Engineering, IIT Roorkee
Enrollment No: 25112057

. Project Overview .

This repository documents my work on frame-level video anomaly detection developed for the Pixel Play competition, and submitted as part of the Vision & Language Group (VLG), IIT Roorkee club recruitment process.
The primary objective of this project was not only to maximize leaderboard performance, but to understand anomaly detection from a metric-driven perspective, with a strong focus on Average Precision (AP) as a global ranking metric.
The final solution is based on a ResNet-18 Autoencoder, trained in a fully unsupervised manner on normal data only, with extensive inference-time experimentation and failure analysis.

. Motivation.

Most anomaly detection pipelines focus on improving reconstruction quality or visual anomaly scores. However, in competitions like Pixel Play, Average Precision depends entirely on global ranking, not on absolute score magnitude.

This project explores:
What actually improves AP
What intuitively seems correct but fails
Where single-frame spatial models reach their practical limits

 . Competition Context.

Competition: Pixel Play
Task: Frame-level anomaly detection
Dataset: Avenue (normal-only training)
Evaluation Metric: Average Precision (AP)
Best Achieved AP: 0.6493
Leaderboard Rank: Near top-10 (final ranks pending)

. Final Model Summary.

Architecture: ResNet-18 Autoencoder
Training:
Normal frames only
Single-frame spatial input
Seed fixed to 82 for reproducibility
Input Resolution: 128 × 128 RGB
Training Frames: 9,204
Loss Function: Mean Squared Error (MSE)
The model intentionally avoids temporal modeling in order to study the maximum achievable performance of spatial autoencoders.

. Training Details .

Optimizer: Adam
Learning Rate: 1e-4
Batch Size: 16
Epochs: 10
Environment: Kaggle Notebook
Training loss converged smoothly from ~0.010 to ~0.0006, indicating stable reconstruction of normal patterns without overfitting.

. Inference Strategy .

Most performance gains came from inference-time design, not architecture changes.
Key techniques used:
Test-Time Augmentation (horizontal flip, vertical flip, 180° rotation)
Pixel-level reconstruction error (L1)
Multi-layer feature consistency error (encoder layer3 and layer4)
Careful score aggregation without per-video normalization
Mild power-law scaling (gamma)
All changes were evaluated strictly based on AP impact, not visual smoothness.

. Key Technical Insights .

Average Precision depends on global ranking across all frames
Per-video normalization consistently degrades AP
Smoothing improves curves but reduces recall at anomaly onset
Noise and compression artifacts dominate Avenue anomalies
Improving minority anomaly types can reduce overall AP

. Failure Analysis (Equal Importance).

Several intuitive ideas were tested and rejected after empirical evaluation:
New architectures (ConvNeXt, ViT-based autoencoders)
Resulted in feature collapse or unstable ranking
Temporal smoothing and post-processing
Reduced AP despite visually cleaner outputs
Orientation-focused heuristics
Improved inversion anomalies but degraded ranking of noise anomalies
Aggressive feature dominance or gamma scaling
Led to ranking collapse and large AP drops
These failures were critical in identifying the practical ceiling of the approach.
Identified Limitations

This project confirms that single-frame spatial autoencoders:

  Capture texture degradation and structural inconsistencies
  Fail to model motion, temporal context, and frequency-domain anomalies
  Further improvement would require:
  Optical flow
  Temporal modeling
  Multi-frame supervision
  These were not pursued due to time constraints and competition scope.

anomaly-detection/
│
├── training_notebook.ipynb
│   └── ResNet-18 autoencoder training (Seed = 82)
│
├── inference_notebook.ipynb
│   └── TTA-based inference, scoring, and submission generation
│
├── models/
│   └── resnet18_autoencoder_seed82.pth
│
├── experiments/
│   └── failure_analysis.md
│
└── README.md

. Conclusion .


This project emphasizes discipline over heuristics and analysis over blind optimization.
The final model represents the best achievable performance within the constraints of a spatial, single-frame setup.
More importantly, the work demonstrates:
Metric-aware thinking
Controlled experimentation
Honest failure analysis
Awareness of model limitations
These lessons are directly transferable to more advanced temporal and multimodal anomaly detection research.
