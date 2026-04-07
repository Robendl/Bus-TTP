# Bus Travel Time Prediction

My MSc thesis for the Artificial Intelligence program at the University of Groningen. Full thesis report can be found [here](thesis-report.pdf).

### Abstract:

Accurate travel time prediction is essential for creating well-designed bus schedules. Existing methods typically depend on GPS trajectories and focus on real-time prediction of known routes, which limits their ability to generalise to unseen routes without historical data. This thesis investigates zero-shot prediction using only static trip attributes and road characteristics. In the Netherlands, such GPS data are not available, so an extensive feature set is constructed to compensate. Four models of increasing complexity were compared: linear regression, XGBoost, a multilayer perceptron (MLP), and a long short-term memory (LSTM) network. Both neural network approaches consistently outperformed the non-neural baselines, with no significant difference between the MLP and the LSTM. Feature importance analysis revealed that distance and maximum speed dominate prediction performance, with other route attributes generally showing greater importance compared to temporal features. These findings indicate the feasibility of travel time prediction in contexts where only static trip attributes and road characteristics are available, thereby supporting transport companies in scheduling decisions.
