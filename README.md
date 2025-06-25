# Self-Interference Cancellation in In-Band Full Duplex Communication

This repository contains the implementation of self-interference (SI) cancellation algorithms for in-band full duplex (IBFD) communication systems. The focus is on advanced non-linear cancellation techniques using:

- **Maximum Versoria Criterion (MVC) Adaptive Algorithm**
- **Neural Networks (NN)**
- **Spline Adaptive Filters (SAF)**

## 📌 Project Overview

In full-duplex systems, simultaneous transmission and reception on the same frequency leads to self-interference. The goal is to cancel this strong interference using robust nonlinear adaptive filters.

### Objectives:
- Mitigate linear and nonlinear self-interference using advanced filtering methods.
- Compare performance between neural network and spline-based MVC algorithms.

## 📈 Techniques Implemented

### 1. MVC Adaptive Filter
- Cancels linear SI using robust outlier-resistant filtering.
- Provides ~37.9 dB of linear SI cancellation.

### 2. MVC + Neural Network
- Deep neural network with 17 ReLU hidden layers.
- Trained using Adam optimizer.
- Handles complex nonlinear residual interference.
- Achieves ~6.46 dB of nonlinear SI cancellation.

### 3. MVC + Spline Adaptive Filter
- Combines FIR filtering with spline-based nonlinear mapping.
- Learns control points for smooth cancellation.
- Achieves ~0.66 dB of nonlinear SI cancellation.

## 🔬 Simulation Results

| Method                  | Linear SI Cancellation | Nonlinear SI Cancellation | Distance from Noise Floor |
|------------------------|------------------------|----------------------------|----------------------------|
| MVC + Neural Network   | 37.90 dB               | 6.46 dB                    | 3.70 dB                    |
| MVC + Spline Filter    | 37.89 dB               | 0.66 dB                    | 9.51 dB                    |

## 📁 Files

- `mvc.py` – Implementation of basic MVC algorithm.
- `fullduplexmvc.py` – End-to-end full-duplex SI cancellation.
- `NNCancellationmvc.py` – MVC with neural network-based SI cancellation.
- `mvcspline.py` – MVC with spline adaptive filter for nonlinear SI cancellation.

## 🧠 Key Insights

- Neural networks show superior non-linear modeling capability due to their deep structure and optimizer-based training.
- Spline filters are lightweight and smooth but less adaptive to abrupt hardware-induced nonlinearities.

## 📚 References

1. Guan & Biswal, *Spline adaptive filtering algorithm based on different iterative gradients*, [Journal of Automation and Intelligence, 2023](https://doi.org/10.1016/j.jai.2022.100008)
2. GeeksforGeeks, *Neural Networks – A Beginner's Guide*, [Link](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
3. Bhattacharjee et al., *Robust adaptive filters using MVC*, IEEE Transactions on Circuits and Systems II, 2021.
4. MathWorks, *Spline Adaptive Filtering Algorithm*, [MATLAB Central](https://in.mathworks.com/matlabcentral/fileexchange/111310-analysis-of-fxlms-based-spline-adaptive-filtering-algorithm)
5. Balatsoukas-Stimming, *Neural SI Cancellation in Full-Duplex Radios*, SPAWC, 2018.

PowerPoint Presentation link:- https://iitgnacin-my.sharepoint.com/:p:/g/personal/23110136_iitgn_ac_in/EdqG3RKEOk1Mg9hZvg32yQkB8RQi_0xdqRWmtfX47duYXQ?e=LgbFmI
