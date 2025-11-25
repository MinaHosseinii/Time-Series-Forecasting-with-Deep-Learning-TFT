# Solar Energy Generation Forecasting for Germany

**Forecasting with Temporal Fusion Transformers (TFT)**

This project focuses on day-ahead, multi-horizon forecasting of solar power generation in Germany using the Temporal Fusion Transformer (TFT) — a state-of-the-art deep learning architecture designed for interpretable sequence modeling. Solar generation is a highly nonlinear time series characterized by strong diurnal cycles, discontinuities between day and night, and sharp ramps at sunrise and sunset. These properties make forecasting challenging, especially for models that assume smooth temporal dynamics. TFT was selected for its ability to integrate attention mechanisms, static covariates, gating layers, and multi-quantile prediction to model uncertainty and temporal dependencies.

**Model Performance and Forecasting Behavior**

The model was evaluated on 30 November 2022 using standard regression metrics, including RMSE, MAE, MSE, and R². Despite the challenging zero-inflated structure of solar data, the TFT produced competitive results with:

RMSE: 692.08 MW

MAE: 347.35 MW

MSE: 478,968 MW²

R²: 0.684

80% prediction interval coverage

The visualizations (see figures in this repository) show that the TFT forecast curve closely follows the true generation profile, particularly during peak daytime hours. However, differences are most visible around sunrise and sunset where transitions are abrupt and nonlinear.

To better handle extended nighttime zero-production, a Weighted Multi-Quantile Loss (WMQLoss) was introduced. This loss function reduces the influence of long zero intervals, allowing the model to prioritize daytime dynamics while still maintaining uncertainty coverage.

**Challenges: Zero-Inflation and Hardware Constraints**

Solar power exhibits a discontinuous time series with predictable zeros during nighttime and sharp daytime variability. These patterns create difficulties for attention-based models like TFT, which rely on rich temporal embeddings. Although TFT includes gating layers and variable selection networks, it still struggled to fully capture:

abrupt transitions from zero to high output

long flat zero regions with low information

rapid fluctuations caused by weather dynamics

In addition, training was affected by GPU memory limitations, which constrained the input sequence length, batch size, and hidden representations. These restrictions limited the model’s ability to learn long-term temporal dependencies and prevented full utilization of the architecture’s high capacity.

**Training Strategy Under Computational Constraints**

To overcome hardware limitations, the dataset (114,000 samples) was divided into multiple training chunks. The model was implemented using PyTorch and the NeuralForecast library, which provided efficient utilities for multi-horizon forecasting and handling temporal covariates. A checkpointing mechanism was implemented to save model weights at regular intervals, enabling training to resume after GPU memory failures or system timeouts.

Although this incremental training process increased overall training time, it ensured:

Successful convergence of the model

Reduced redundant computation by avoiding retraining from scratch

Robust recovery from crashes

Complete coverage of the full training dataset

This adaptive pipeline allowed the Temporal Fusion Transformer to achieve stable and reliable performance, even under limited computational resources, while leveraging the high-capacity architecture provided by PyTorch and NeuralForecast.

**Key Outcomes**

Implemented and trained a Temporal Fusion Transformer for multi-horizon solar forecasting

Developed a custom WMQLoss function to handle zero-inflated data

Built a scalable, fault-tolerant training pipeline using chunking and checkpointing

Evaluated performance with industry-standard metrics and uncertainty quantification

Demonstrated the practical challenges of applying large attention-based models to nonlinear, discontinuous time series under real-world hardware constraints
