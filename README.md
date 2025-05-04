Project Explanation: Predicting Pokémon Legendary Status
Objective
Build a machine learning model to predict whether a Pokémon is Legendary (e.g., Mewtwo, Lugia) based on its attributes, such as:

Stats (Health, Attack, Defense, etc.)

Typing (Primary/Secondary types like Fire, Water, Psychic)

Generation (Which game series it belongs to)

Physical traits (Height, Weight)

1. Data Preparation
Missing Values: Pokémon without a secondary type have NaN in the Secondary Typing column. These are replaced with "None" to explicitly indicate no secondary type.

Categorical Encoding: Convert text-based features (e.g., Primary Typing, Generation) into numerical values using one-hot encoding. For example:

Primary Typing_Fire = 1 (if the Pokémon is Fire-type).

Secondary Typing_None = 1 (if the Pokémon has no secondary type).

Irrelevant Features: Remove non-predictive columns like Name or National Dex #, which don’t influence Legendary status.

2. Model Architecture
A neural network is built using TensorFlow with:

Input Layer: Accepts the processed features (e.g., stats, encoded types).

Hidden Layers: Multiple dense layers with ReLU activation to learn complex patterns.

Example: Dense(128, activation='relu') (128 neurons in a layer).

Dropout Layers: Prevent overfitting by randomly disabling neurons during training.

Output Layer: A single neuron with sigmoid activation to predict the probability of being Legendary (0 = Not Legendary, 1 = Legendary).

3. Handling Class Imbalance
Legendary Pokémon are rare (e.g., only ~5% of the dataset). To avoid bias toward non-Legendary Pokémon:

Class Weights: Assign higher weight to Legendary samples during training.

SMOTE: Artificially generate synthetic Legendary Pokémon samples to balance the dataset.

4. Training & Evaluation
Training: The model learns patterns from the training data using the Adam optimizer and binary cross-entropy loss.

Evaluation Metrics:

Precision: How many predicted Legendary Pokémon are actually Legendary.

Recall: How many actual Legendary Pokémon are correctly identified.

Confusion Matrix: Visualize true/false positives/negatives.
