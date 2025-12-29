Model Card: PD-Mechanistic-Interpreter-v1
1. Model Details
Developer: [Your Name]

Model Date: December 2025

Model Type: Multi-Layer Perceptron (MLP) Classifier with a post-hoc Sparse Autoencoder (SAE) for mechanistic interpretability.

Domain: Healthcare / Neurology (Parkinson’s Disease).

Framework: PyTorch 2.x.

2. Intended Use
Primary Purpose: To detect Parkinson’s Disease using vocal biomarkers and provide a human-interpretable "reasoning path" for the diagnosis.

Target Users: AI Researchers, Neurologists, and Clinical Auditors.

Out of Scope: This model is a research prototype. It is not an FDA-cleared diagnostic tool and should not be used for making independent clinical decisions.

3. Factors
Demographic Factors: The model’s performance may vary based on a patient's age, gender, and native language (due to different vocal baseline frequencies).

Technical Factors: Microphone quality, background noise, and sampling rate (Hz) significantly impact the "shimmer" and "jitter" feature extractions.

4. Metrics
Primary Metric: Classification Accuracy.

Interpretability Metric: Reconstruction Fidelity (how well the SAE mirrors the original model) and Sparsity (L1 norm of latent activations).

Causal Metric: Ablation Accuracy Drop (measuring the necessity of specific latent circuits).

5. Training & Evaluation Data
Dataset: Oxford Parkinson's Disease Detection Dataset.

Splits: 80% Training / 20% Testing.

Preprocessing: Standard scaling of 22 biomedical voice features.

6. Quantitative Analysis (Ablation Results)
Baseline Accuracy: 94.87%

Intervention: Surgically silenced Latent Feature #120 (Vocal Dysphonia Circuit).

Post-Intervention Accuracy: 87.18%

Impact: A 7.69% drop, proving the feature is a necessary component for high-fidelity diagnosis.

7. Ethical Considerations & Safety
Alignment: The model's decision-making is aligned with clinical pathology (Dysphonia markers) rather than spurious proxies (like patient ID or recording date).

Bias: Further research is required to ensure Feature #120 remains robust across diverse accents and respiratory conditions (e.g., asthma).

Privacy: No raw audio data is stored; only anonymized biomedical features are used for inference.

8. Caveats & Recommendations
Redundancy: The model still retains 87% accuracy without its primary feature, suggesting it uses multiple redundant paths for diagnosis. Future research should map the "secondary" circuits.

Recommendation: Use this model as a "Second Opinion" tool for clinicians to visualize vocal instability patterns.
