# README.md

## Phase 1: Define the Input for Inference

**Objective:** Understand and formalize the problem statement to define the input space for the neural network.

### Topics to Learn & Implement
- **Understanding Path Planning Fundamentals**
    - Classic path planning algorithms: A*, RRT*, Hybrid A*
    - Trajectory optimization methods: Convex optimization, MPC, Pontryagin’s Minimum Principle
    - Cost functions for optimal path selection
- **Problem Formulation for Neural Networks**
    - What information is needed as input?
    - What constraints are essential?
    - How should the output be represented?

### Deliverables
- ✅ Formally define the input features for inference (e.g., map representation, obstacle positions, dynamic constraints).
- ✅ Decide on the output format (e.g., control points of a B-spline trajectory).
- ✅ Write a document summarizing the problem formulation.

---

## Phase 2: Synthesize Valid Inputs (Data Generation)

**Objective:** Create a simulation environment or dataset generator to produce diverse path planning scenarios.

### Topics to Learn & Implement
- **Generating Synthetic Scenarios**
    - Environment representation (grid maps, vector maps, occupancy grids)
    - Generating random obstacles and traffic conditions
    - Handling dynamic obstacles
- **Using a Simulator for Data Generation**
    - CARLA or SUMO for autonomous driving scenarios
    - Custom Python environment for synthetic data

### Deliverables
- ✅ A Python script that generates diverse scenarios for training.
- ✅ Define variations in environment conditions (narrow roads, intersections, moving obstacles).

---

## Phase 3: Solve Each Frame Using a Mathematical Approach

**Objective:** Implement an optimization-based solver to generate ground truth paths.

### Topics to Learn & Implement
- **Trajectory Optimization Techniques**
    - Quadratic programming (QP) for smooth paths
    - Model Predictive Control (MPC) formulation for dynamic constraints
    - CasADi for non-linear optimization
- **Validating and Evaluating Solutions**
    - Metrics: Path length, smoothness, feasibility
    - Sensitivity analysis

### Deliverables
- ✅ A solver that computes an optimal trajectory given an input frame.
- ✅ Store computed solutions as labeled training data.

---

## Phase 4: Prepare a DNN Architecture

**Objective:** Design a neural network that can learn to imitate the optimal solver.

### Topics to Learn & Implement
- **Choosing the Right Architecture**
    - CNNs for image-based inputs
    - Transformers for spatial reasoning
    - Graph Neural Networks (GNNs) for road networks
- **Loss Function Design**
    - Supervised learning using trajectory optimization as ground truth
    - Regression loss vs. reinforcement learning-based reward shaping
- **Network Training**
    - Data augmentation techniques
    - Optimization techniques (Adam, learning rate scheduling)

### Deliverables
- ✅ A defined neural network architecture that is trainable on the synthetic dataset.
- ✅ Initial experiments to validate the network’s ability to approximate optimal solutions.

---

## Phase 5: Training & Validation

**Objective:** Train the network and evaluate its performance against the ground truth solver.

### Topics to Learn & Implement
- **Training Setup**
    - Data pipeline for training and validation splits
    - Hyperparameter tuning
- **Evaluation Metrics**
    - Loss convergence analysis
    - Comparison of predicted vs. optimal trajectories

### Deliverables
- ✅ A trained network with performance metrics and validation results.
- ✅ Visualization of predicted vs. optimal trajectories.

---

## Phase 6: Deployment in C++

**Objective:** Convert the trained network into a deployable format and integrate it into a C++ pipeline.

### Topics to Learn & Implement
- **Exporting the Model**
    - ONNX or TensorRT conversion for C++ inference
    - Quantization and model compression
- **Integration with C++**
    - Writing a C++ wrapper for inference
    - Real-time performance testing

### Deliverables
- ✅ A deployable version of the trained network.
- ✅ A C++ implementation that loads the model and performs inference.

---

## Final Outcome

🎯 A fully functional pipeline that:
- Generates synthetic driving scenarios.
- Computes optimal paths using a solver.
- Trains a deep learning model to imitate the solver.
- Deploys the trained model in a C++ environment.

---

## Estimated Timeline

| Phase | Task                     | Estimated Time       |
|-------|--------------------------|----------------------|
| 1     | Define input format      | 1 week              |
| 2     | Generate synthetic data  | 2 weeks             |
| 3     | Implement solver         | 3-4 weeks           |
| 4     | Design DNN architecture  | 2 weeks             |
| 5     | Train and validate       | 4 weeks             |
| 6     | Deploy in C++            | 2-3 weeks           |

**Total:** 3-4 months (depending on experience with ML and C++ deployment)
