<div style="display: flex; align-items: center; gap: 10px;">
  <img src="../Images/Poppy_Universe_Logo.png" alt="Poppy Universe Logo" width="100" style="margin-top: -5px;"/>
  <h1>Poppy Universe - Machine Learning</h1>
</div>

> **Simulated datasets, MF & NN predictions, ready for the engine**

This folder contains all **Machine Learning code** for the **Poppy Universe** project.  
It goes from **user interaction data** to meaningful insights and predictions for the recommendation engine.

---

## 🚀 Purpose

* Simulate user interactions (views, clicks, favorites) for moons, planets, and stars (different approaches for each layer).  
* Compute **Layer 2 liking scores** per object.  
* Run **Matrix Factorization (Layer 3)** and **Neural Network (Layer 4)** for category-level predictions.  
* Generate outputs compatible for the recommendation engine.  
* Use **hardcoded rules for simulations**, but once enough real data is collected, the same notebooks can process **actual user interactions**.

---

## 📂 Project Structure

```tree
Machine_Learning/
├── Data_Prep/                                  # Notebooks to create simulated datasets
│   └── Data_Creation_Layer_x.ipynb
├── Input_Data/                                 # Raw & simulated datasets
│   ├── MF_Sematnic_Type_Interactions.csv           # Layer 3
│   ├── NN_Semantic_Interactions.csv                # Layer 4
│   └── Simulated_User_Interactions.csv             # Layer 2
├── Models/                                     # MF and NN notebooks per layer
│   ├── Layer2/                                     # Layer 2 notebooks
│   │   └── Layer2_User_Scores.ipynb                    # Calculates object liking scores
│   ├── Layer3/                                     # Layer 3 notebooks
│   │   ├── Files                                       # Temp data diles
│   │   ├── Layer3_MF_Moons.ipynb                       # Moons notebook
│   │   ├── Layer3_MF_Planets.ipynb                     # Planets notebook
│   │   ├── Layer3_MF_Stars.ipynb                       # Stars notebook
│   │   └── Layer3_Master.ipynb                         # Combines output data from layer 3 notebooks.
│   ├── Layer4/                                     # Layer 4 notebooks
│   │   ├── Files
│   │   ├── Layer4_NN_Moons.ipynb                       # Moons notebook
│   │   ├── Layer4_NN_Planets.ipynb                     # Planets notebook
│   │   ├── Layer4_NN_Stars.ipynb                       # Stars notebook
│   │   └── Layer4_Master.ipynb                         # Combines output data from layer 4 notebooks.
│   └── Plots/                                      # visualizations
├── Output_Data/                                # Prediction outputs for the engine
└── README.md                                   # This README
```

## 🏗️ Layer Explanations

### 🌓 Layer 2 — Object Liking Scores

> Layer 2 simulates **user interactions** and calculates a **liking score per object**.  
> These scores can be used as input for Layer 3 MF and Layer 4 NN models.

- **Inputs used:** 
  - Output from layer 1 
  - Simulated user interactions (views, clicks, favorites)  
  - Object types: Moons, planets, stars  

- **How it works:**  
  1. Aggregates interactions per user × object.  
  2. Computes a **total liking score** combining views, clicks, favorites.  
  3. Normalizes scores to create a consistent input for ML models.

- **Returns:**  
  - CSV with object-level scores: `Object_Type,Object_ID,total_interactions,num_views,num_clicks,num_favorites,trending_score`  

- **Note:**  
  - Rules are **hardcoded for simulation**.  
  - Once enough real data exists, these can be replaced with actual interactions.

### 🚀 Layer 3 — Matrix Factorization

> Layer 3 is the **category-level prediction layer using MF**.  
> Focuses on **semantic patterns** across categories (star types, planet types, moon parents).

- **Inputs used:**  
  - Ouptput from layer 1  
  - User × category matrices for stars, planets, moons  

- **How it works:**  
  1. Builds **User × Category matrices** (rows = users, columns = categories).  
  2. Fills missing interactions with 0, optionally normalizes.  
  3. Performs **matrix factorization** to extract latent features.  
  4. Predicts missing interactions, producing **user × category scores**.

- **Returns:**  
  - CSV with predicted scores: `User_ID,A,B,F,G,K,M,O,Dwarf Planet,Gas Giant,Ice Giant,Terrestrial,Earth,...`  
  - Used to rank categories for each user or as input for Layer 4.

- **Notes:**  
  - Simulated input is regenerated each run to get slightly different data.  
  - Hardcoded rules apply for now; real data can replace it once validated.

### 🌠 Layer 4 — Neural Network

> Layer 4 refines predictions using a **from-scratch neural network**.  
> Captures **nonlinear patterns** and interactions between users and categories.

- **Inputs used:**  
  - Output from layer 1
  - Simulated or real user × category data  
  - One-hot encoding for users and categories  
  - Interaction strength as target labels  

- **How it works:**  
  1. Encodes inputs for the NN.  
  2. Forward pass computes predicted scores through hidden layers with **tanh activations**.  
  3. Loss calculation against actual interaction strengths.  
  4. Backpropagation updates weights and biases with gradient descent.  
  5. Trains for multiple epochs; mini-batches optional.  
  6. Produces predicted scores for all user × category combinations.

- **Returns:**  
  - Refined **user × category predictions**  
  - CSV output for integration with the recommendation engine

- **Notes:**  
  - NN input is **simulated each run** for variety.  
  - Hardcoded rules currently define initial inputs; can be replaced with real interactions once verified.

---

## ⚙️ Usage Notes

* Simulated data allows testing MF and NN pipelines **before real user data exists**.  
* Each notebook can be run standalone for testing, or as part of the **ML workflow for the engine**.  
* Master notebooks (Layer3_Master & Layer4_Master) check if enough real data exists; otherwise, they default to simulated datasets.

---

## 🌠 Outputs

* **Object-level liking scores** (Layer 2)  
* **Predicted category-level scores** (Layer 3 MF)  
* **Refined category predictions** (Layer 4 NN)  
* All outputs saved as **CSV files** in `Output_Data/` for engine integration.

---

## 🌟 Future Plans

* Add **Business Logic** (Layer 5).
* Fully integrate with backend, frontend, and ML modules.
* Turn this into the **complete Poppy Universe project repo**, containing engine, frontend, backend, data, and ML.

---

## 🛠 Author

**Jelle Smet**



<p align="center">
  <img src="../Images/Poppy_Universe_Logo.png" alt="Poppy Universe Logo" width="600"/>
</p>